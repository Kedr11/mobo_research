import warnings

import gpytorch
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior

try:
    from botorch.acquisition.logei import qLogExpectedImprovement
except ImportError:
    qLogExpectedImprovement = None

try:
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
except ImportError:
    qLogNoisyExpectedImprovement = None


class BotorchRunner:
    def __init__(self, problem, acq_type="qEHVI", device=None, dtype=torch.float64, verbose=False, keep_history=False):
        self.problem = problem
        self.acq_type = acq_type
        self.device = device if device else torch.device("cpu")
        self.dtype = dtype
        self.verbose = verbose
        self.keep_history = keep_history

        self.train_x = torch.empty(0, problem.dim, device=self.device, dtype=self.dtype)
        self.train_y = torch.empty(0, problem.num_objectives, device=self.device, dtype=self.dtype)
        self.bounds = problem.bounds.to(device=self.device, dtype=self.dtype)

        self.parego_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([192]))
        ehvi_sample_count = 96 if problem.num_objectives >= 3 else 128
        self.qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([ehvi_sample_count]))
        self.num_restarts = 12 if problem.num_objectives >= 3 else 20
        self.raw_samples = 384 if problem.num_objectives >= 3 else 1024
        self.opt_options = {
            "batch_limit": 4 if problem.num_objectives >= 3 else 5,
            "maxiter": 100 if problem.num_objectives >= 3 else 200,
        }
        self.qehvi_num_restarts = 12 if problem.num_objectives >= 3 else 16
        self.qehvi_raw_samples = 256 if problem.num_objectives >= 3 else 384
        self.qehvi_opt_options = {"batch_limit": 4, "maxiter": 100}
        self._qehvi_ic_pool_size = 512 if problem.num_objectives >= 3 else 768

        self._parego_weight_bank = self._build_parego_weight_bank(bank_size=128)
        self._parego_weight_index = 0
        self._parego_probe_points = 64 if problem.num_objectives >= 3 else 128
        self._parego_ic_pool_size = 512 if problem.num_objectives >= 3 else 2048
        self._parego_max_attempts = 4 if problem.num_objectives >= 3 else 8
        if qLogNoisyExpectedImprovement is not None:
            self._parego_acq_kind = "qLogNEI"
        elif qLogExpectedImprovement is not None:
            self._parego_acq_kind = "qLogEI"
        elif qNoisyExpectedImprovement is not None:
            self._parego_acq_kind = "qNEI"
        else:
            self._parego_acq_kind = "qEI"

        self._parego_uses_nei = self._parego_acq_kind in {"qLogNEI", "qNEI"}
        self._parego_uses_logei = self._parego_acq_kind in {"qLogNEI", "qLogEI"}
        self._last_acq_info = {}
        self.last_iteration_info = {}
        self.history = [] if keep_history else None
        self.iteration_index = 0

    def _log(self, message):
        if self.verbose:
            print(message)

    def initialize_data(self, n=10):
        lower = self.bounds[0]
        upper = self.bounds[1]
        span = upper - lower

        if self.problem.family == "DTLZ" or self.problem.name == "zdt4":
            center = (lower + upper) / 2.0
            anchor_points = [center]

            for boundary_value in (lower[0], upper[0]):
                point = center.clone()
                point[0] = boundary_value
                anchor_points.append(point)

            max_aux_dims = min(self.problem.dim - 1, 4)
            for dim_idx in range(1, max_aux_dims + 1):
                delta = 0.2 * span[dim_idx]
                for direction in (-1.0, 1.0):
                    point = center.clone()
                    point[dim_idx] = torch.clamp(center[dim_idx] + direction * delta, lower[dim_idx], upper[dim_idx])
                    anchor_points.append(point)

            anchors = torch.stack(anchor_points, dim=0)
            sobol_center = center
            sobol_radius = 0.25 * span
            sobol_bounds = torch.stack(
                [
                    torch.maximum(lower, sobol_center - sobol_radius),
                    torch.minimum(upper, sobol_center + sobol_radius),
                ]
            )
        else:
            anchors = torch.cat(
                [
                    lower.unsqueeze(0),
                    lower.unsqueeze(0) + torch.eye(self.problem.dim, device=self.device, dtype=self.dtype) * span,
                ],
                dim=0,
            )
            sobol_bounds = self.bounds

        if n <= anchors.shape[0]:
            new_x = anchors[:n]
        else:
            sobol_x = draw_sobol_samples(bounds=sobol_bounds, n=n - anchors.shape[0], q=1).squeeze(1)
            new_x = torch.cat([anchors, sobol_x], dim=0)

        new_y = self.problem.evaluate(new_x).to(device=self.device, dtype=self.dtype)

        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])

    def _build_parego_weight_bank(self, bank_size):
        if self.problem.num_objectives == 2:
            grid = torch.linspace(1e-3, 1.0 - 1e-3, steps=bank_size, device=self.device, dtype=self.dtype)
            ordered = []
            left = 0
            right = bank_size - 1
            while left <= right:
                ordered.append(grid[left])
                if left != right:
                    ordered.append(grid[right])
                left += 1
                right -= 1

            first_objective_weights = torch.stack(ordered[:bank_size])
            weights = torch.stack(
                [first_objective_weights, 1.0 - first_objective_weights],
                dim=-1,
            )
            return weights / weights.sum(dim=-1, keepdim=True)

        weights = sample_simplex(
            d=self.problem.num_objectives,
            n=bank_size,
            qmc=True,
            seed=0,
            device=self.device,
            dtype=self.dtype,
        )

        weights = weights.clamp_min(1e-3)
        return weights / weights.sum(dim=-1, keepdim=True)

    def _next_parego_weights(self, n_points):
        bank_size = self._parego_weight_bank.shape[0]
        indices = (torch.arange(n_points, device=self.device) + self._parego_weight_index) % bank_size
        self._parego_weight_index = int((self._parego_weight_index + n_points) % bank_size)
        return self._parego_weight_bank[indices]

    def _get_valid_train_y(self):
        valid_y = self.train_y[~torch.isnan(self.train_y).any(dim=1)]
        if valid_y.numel() == 0:
            raise RuntimeError("No valid objective values are available.")
        return valid_y

    def _get_pareto_train_y(self):
        valid_y = self._get_valid_train_y()
        if valid_y.shape[0] <= 1:
            return valid_y

        is_efficient = torch.ones(valid_y.shape[0], dtype=torch.bool, device=valid_y.device)
        for i in range(valid_y.shape[0]):
            if not is_efficient[i]:
                continue
            dominates_i = (valid_y >= valid_y[i]).all(dim=1) & (valid_y > valid_y[i]).any(dim=1)
            if dominates_i.any():
                is_efficient[i] = False
        return valid_y[is_efficient]

    def _get_parego_scalarization(self, weights, model=None):
        if self._parego_uses_nei:
            if model is None:
                raise RuntimeError("ParEGO scalarization for NEI requires a fitted model.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                baseline_y = model.posterior(self.train_x).mean
            scalarization = get_chebyshev_scalarization(weights=weights, Y=baseline_y)
            return scalarization, baseline_y

        valid_y = self._get_valid_train_y()
        pareto_y = self._get_pareto_train_y()
        scalarization_y = pareto_y if pareto_y.shape[0] >= 2 else valid_y
        scalarization = get_chebyshev_scalarization(weights=weights, Y=scalarization_y)
        return scalarization, valid_y

    def _get_fitted_model(self):
        models = []
        for i in range(self.problem.num_objectives):
            train_y_slice = self.train_y[:, i : i + 1]
            target_std = train_y_slice.std(dim=0, unbiased=False).clamp_min(1.0)
            # Keep the effective noise floor stable after Standardize().
            train_yvar = torch.full_like(train_y_slice, 1e-4) * target_std.pow(2)

            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.problem.dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )

            gp = SingleTaskGP(
                train_X=self.train_x,
                train_Y=train_y_slice,
                train_Yvar=train_yvar,
                input_transform=Normalize(d=self.problem.dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
                covar_module=covar_module,
            )
            models.append(gp)

        model_list = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)

        with warnings.catch_warnings(), gpytorch.settings.cholesky_jitter(1e-4):
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)

        return model_list

    def _build_parego_acq_func(self, model, weights):
        scalarization, baseline_y = self._get_parego_scalarization(weights, model=model)
        objective = GenericMCObjective(scalarization)

        if self._parego_acq_kind == "qLogNEI":
            acq_func = qLogNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=self.train_x,
                sampler=self.parego_sampler,
                prune_baseline=True,
                cache_root=False,
            )
            return acq_func, None

        if self._parego_acq_kind == "qNEI":
            acq_func = qNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=self.train_x,
                sampler=self.parego_sampler,
                prune_baseline=True,
                cache_root=False,
            )
            return acq_func, None

        best_f = scalarization(baseline_y).max()
        acq_cls = qLogExpectedImprovement if self._parego_acq_kind == "qLogEI" else qExpectedImprovement
        acq_func = acq_cls(model=model, objective=objective, best_f=best_f, sampler=self.parego_sampler)
        return acq_func, best_f

    def _probe_acq_value(self, acq_func):
        probe_x = draw_sobol_samples(bounds=self.bounds, n=self._parego_probe_points, q=1).squeeze(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe_values = acq_func(probe_x.unsqueeze(-2))

        fill_value = -1e12 if self._parego_uses_logei else 0.0
        probe_values = torch.nan_to_num(
            probe_values.reshape(-1),
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value,
        )
        return probe_values.max().item()

    def _get_parego_initial_conditions(self, acq_func):
        pool_x = draw_sobol_samples(bounds=self.bounds, n=self._parego_ic_pool_size, q=1).squeeze(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pool_values = acq_func(pool_x.unsqueeze(-2)).reshape(-1)

        fill_value = -1e12
        pool_values = torch.nan_to_num(
            pool_values,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value,
        )

        topk = min(self.num_restarts, pool_values.shape[0])
        top_indices = torch.topk(pool_values, k=topk).indices
        batch_initial_conditions = pool_x[top_indices].unsqueeze(-2)
        best_candidate = pool_x[top_indices[0]].unsqueeze(0)
        best_value = pool_values[top_indices[0]].item()
        return batch_initial_conditions, best_candidate, best_value

    def _get_parego_fallback_candidate(self, model, weights):
        pool_x = draw_sobol_samples(bounds=self.bounds, n=self._parego_ic_pool_size, q=1).squeeze(1)
        scalarization, _ = self._get_parego_scalarization(weights, model=model)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posterior = model.posterior(pool_x)
            optimistic_y = posterior.mean + 0.2 * posterior.variance.clamp_min(0.0).sqrt()
            scores = scalarization(optimistic_y).reshape(-1)

        scores = torch.nan_to_num(scores, nan=-1e12, posinf=-1e12, neginf=-1e12)
        best_index = scores.argmax()
        return pool_x[best_index : best_index + 1], scores[best_index : best_index + 1]

    def _get_qehvi_initial_conditions(self, acq_func, n_points):
        pool_x = draw_sobol_samples(bounds=self.bounds, n=self._qehvi_ic_pool_size, q=n_points)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pool_values = acq_func(pool_x).reshape(-1)

        pool_values = torch.nan_to_num(pool_values, nan=0.0, posinf=0.0, neginf=0.0)
        topk = min(self.qehvi_num_restarts, pool_values.shape[0])
        top_indices = torch.topk(pool_values, k=topk).indices
        batch_initial_conditions = pool_x[top_indices]
        best_candidate = pool_x[top_indices[0]]
        best_value = pool_values[top_indices[0]].item()
        return batch_initial_conditions, best_candidate, best_value

    def _get_acq_func(self, model, n_points=1):
        if self.acq_type == "qEHVI":
            with torch.no_grad():
                valid_y = self._get_valid_train_y()
                pareto_y = self._get_pareto_train_y()
                y_reference = pareto_y if pareto_y.shape[0] > 0 else valid_y
                default_ref_point = self.problem.get_ref_point().to(device=self.device, dtype=self.dtype)
                y_min = y_reference.min(dim=0).values
                y_max = y_reference.max(dim=0).values
                span = (y_max - y_min).clamp_min(1e-3)
                adaptive_ref_point = y_min - 0.1 * span
                ref_point = torch.minimum(default_ref_point, adaptive_ref_point)

                better_than_ref = (y_reference > ref_point).all(dim=1)
                y_for_partitioning = y_reference[better_than_ref] if better_than_ref.any() else y_reference
                partitioning = NondominatedPartitioning(ref_point=ref_point, Y=y_for_partitioning)

            self._last_acq_info = {"name": "qEHVI"}
            return (
                qExpectedHypervolumeImprovement(
                    model=model,
                    ref_point=ref_point,
                    partitioning=partitioning,
                    sampler=self.qehvi_sampler,
                ),
                self._last_acq_info,
            )

        if self.acq_type != "ParEGO":
            raise ValueError(f"Unsupported acquisition type: {self.acq_type}")

        acq_func_list = []
        weights_used = []
        best_f_values = []
        probe_values = []

        for _ in range(n_points):
            best_choice = None
            for _ in range(self._parego_max_attempts):
                weights = self._next_parego_weights(1).squeeze(0)
                acq_func, best_f = self._build_parego_acq_func(model, weights)
                probe_max = self._probe_acq_value(acq_func)
                candidate = (probe_max, weights, best_f, acq_func)

                if best_choice is None or probe_max > best_choice[0]:
                    best_choice = candidate
                if torch.isfinite(torch.tensor(probe_max)):
                    break

            probe_max, weights, best_f, acq_func = best_choice
            acq_func_list.append(acq_func)
            weights_used.append(weights.detach().cpu())
            best_f_values.append(float(best_f.detach().cpu()) if best_f is not None else None)
            probe_values.append(probe_max)

        self._last_acq_info = {
            "name": "ParEGO",
            "acq_label": self._parego_acq_kind,
            "weights": weights_used,
            "best_f": best_f_values,
            "probe_max": probe_values,
        }

        if n_points == 1:
            return acq_func_list[0], self._last_acq_info
        return acq_func_list, self._last_acq_info

    def _optimize_candidates(self, model, acq_func, n_points):
        with gpytorch.settings.cholesky_jitter(1e-4):
            if self.acq_type == "ParEGO":
                if n_points == 1:
                    weights = self._last_acq_info["weights"][0].to(device=self.device, dtype=self.dtype)
                    batch_initial_conditions, best_candidate, best_value = self._get_parego_initial_conditions(acq_func)
                    if best_value <= -1e11:
                        return self._get_parego_fallback_candidate(model, weights)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            return optimize_acqf(
                                acq_function=acq_func,
                                bounds=self.bounds,
                                q=1,
                                num_restarts=batch_initial_conditions.shape[0],
                                options=self.opt_options,
                                batch_initial_conditions=batch_initial_conditions,
                            )
                    except RuntimeError:
                        return self._get_parego_fallback_candidate(model, weights)
                return optimize_acqf_list(
                    acq_function_list=acq_func,
                    bounds=self.bounds,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    options=self.opt_options,
                )

            return optimize_acqf(
                acq_function=acq_func,
                bounds=self.bounds,
                q=n_points,
                num_restarts=self.qehvi_num_restarts,
                raw_samples=self.qehvi_raw_samples,
                options=self.qehvi_opt_options,
                sequential=n_points > 1,
            )

    def _optimize_qehvi_candidates(self, acq_func, n_points):
        batch_initial_conditions, best_candidate, best_value = self._get_qehvi_initial_conditions(acq_func, n_points)
        if best_value <= 0.0:
            return best_candidate, torch.tensor([best_value], device=self.device, dtype=self.dtype)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return optimize_acqf(
                    acq_function=acq_func,
                    bounds=self.bounds,
                    q=n_points,
                    num_restarts=batch_initial_conditions.shape[0],
                    options=self.qehvi_opt_options,
                    batch_initial_conditions=batch_initial_conditions,
                    sequential=n_points > 1,
                )
        except RuntimeError:
            return best_candidate, torch.tensor([best_value], device=self.device, dtype=self.dtype)

    def run_iteration(self, n_points=1):
        model = self._get_fitted_model()
        acq_func, acq_info = self._get_acq_func(model, n_points=n_points)
        if self.acq_type == "qEHVI":
            candidates, acq_values = self._optimize_qehvi_candidates(acq_func, n_points=n_points)
        else:
            candidates, acq_values = self._optimize_candidates(model, acq_func, n_points=n_points)

        if isinstance(acq_values, torch.Tensor):
            fill_value = -1e12 if self._parego_uses_logei else 0.0
            max_acq_val = torch.nan_to_num(
                acq_values.reshape(-1),
                nan=fill_value,
                posinf=fill_value,
                neginf=fill_value,
            ).max().item()
        else:
            max_acq_val = float(acq_values)

        if self.acq_type == "ParEGO":
            weights_str = ", ".join(
                "[" + ", ".join(f"{value:.3f}" for value in weights.tolist()) + "]"
                for weights in acq_info["weights"]
            )
            probe_str = ", ".join(f"{value:.2e}" for value in acq_info["probe_max"])
            self._log(f"ParEGO weights: {weights_str}")
            self._log(f"ParEGO acquisition: {acq_info['acq_label']}")
            self._log(f"ParEGO probe acq max: {probe_str}")

        if (not self._parego_uses_logei and max_acq_val < 1e-8) or not torch.isfinite(torch.tensor(max_acq_val)):
            self._log(f"Warning: acquisition max is nearly zero ({max_acq_val:.2e}).")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            posterior = model.posterior(candidates)
            posterior_mean = posterior.mean.detach().squeeze()
            posterior_std = posterior.variance.sqrt().detach().squeeze()
            self._log(f"Model Mean: {posterior_mean}")
            self._log(f"Model Std:  {posterior_std}")

        new_y = self.problem.evaluate(candidates).to(device=self.device, dtype=self.dtype)

        self.train_x = torch.cat([self.train_x, candidates])
        self.train_y = torch.cat([self.train_y, new_y])
        self.iteration_index += 1

        self.last_iteration_info = {
            "iteration": self.iteration_index,
            "acq_type": self.acq_type,
            "acq_value": float(max_acq_val),
            "acq_label": acq_info.get("acq_label", self.acq_type),
            "candidate_x": candidates.detach().cpu().reshape(-1).tolist(),
            "new_y_model_space": new_y.detach().cpu().reshape(-1).tolist(),
            "new_y_minimization": (-new_y).detach().cpu().reshape(-1).tolist(),
            "posterior_mean": posterior_mean.detach().cpu().reshape(-1).tolist() if isinstance(posterior_mean, torch.Tensor) else [float(posterior_mean)],
            "posterior_std": posterior_std.detach().cpu().reshape(-1).tolist() if isinstance(posterior_std, torch.Tensor) else [float(posterior_std)],
            "train_size": int(self.train_x.shape[0]),
        }

        if self.acq_type == "ParEGO":
            self.last_iteration_info["parego_weights"] = [
                [float(value) for value in weights.tolist()]
                for weights in acq_info.get("weights", [])
            ]
            self.last_iteration_info["parego_probe_max"] = [float(value) for value in acq_info.get("probe_max", [])]

        if self.keep_history:
            self.history.append(self.last_iteration_info.copy())

        del model
        del acq_func
        del acq_values
        del posterior
        del posterior_mean
        del posterior_std

        return candidates, new_y
