import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

# Импорты стратегий выбора точек
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.sampling.normal import SobolQMCNormalSampler
# ВАЖНО: Новый импорт для BoTorch 0.17.2
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning

class BotorchRunner:
    def __init__(self, problem, acq_type="qEHVI", device=None, dtype=torch.float64):
        """
        Инициализация 'двигателя' оптимизации.
        """
        self.problem = problem
        self.acq_type = acq_type
        self.device = device if device else torch.device("cpu")
        self.dtype = dtype

        # Списки тензоров
        self.train_x = torch.empty(0, problem.dim, device=self.device, dtype=self.dtype)
        self.train_y = torch.empty(0, problem.num_objectives, device=self.device, dtype=self.dtype)

        # Границы задачи
        self.bounds = problem.bounds.to(device=self.device, dtype=self.dtype)

    def initialize_data(self, n=10):
        """
        ШАГ 1: Разведка начальными точками.
        """
        from botorch.utils.sampling import draw_sobol_samples

        new_x = draw_sobol_samples(bounds=self.bounds, n=n, q=1).squeeze(1)
        new_y = self.problem.evaluate(new_x).to(device=self.device, dtype=self.dtype)

        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])

    def _get_fitted_model(self):
        """
        ШАГ 2: Построение суррогата (модели мира).
        """
        models = []
        for i in range(self.problem.num_objectives):
            train_y_slice = self.train_y[:, i: i + 1]

            gp = SingleTaskGP(
                train_X=self.train_x,
                train_Y=train_y_slice,
                input_transform=Normalize(d=self.problem.dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1)
            )
            models.append(gp)

        model_list = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
        fit_gpytorch_mll(mll)
        return model_list

    def _get_acq_func(self, model):
        """
        ШАГ 3: Стратегия выбора.
        """
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        if self.acq_type == "qEHVI":
            ref_point = self.problem.get_ref_point().to(device=self.device, dtype=self.dtype)

            # ВАЖНО: Явное создание объекта разбиения для BoTorch 0.17.2
            # Используем torch.no_grad(), так как разбиение не требует градиентов
            with torch.no_grad():
                partitioning = NondominatedPartitioning(
                    ref_point=ref_point,
                    Y=self.train_y
                )

            return qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                partitioning=partitioning,  # Передаем обязательный аргумент
                sampler=sampler,
            )

        elif self.acq_type == "ParEGO":
            weights = torch.randn(self.problem.num_objectives, device=self.device, dtype=self.dtype).abs()
            weights /= weights.sum()

            post_transform = ScalarizedPosteriorTransform(weights=weights)

            with torch.no_grad():
                scalarized_y = self.train_y @ weights
                best_f = scalarized_y.max()

            return qExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler,
                posterior_transform=post_transform
            )

    def run_iteration(self, n_points=1):
        """
        ШАГ 4: Цикл действия.
        """
        model = self._get_fitted_model()
        acq_func = self._get_acq_func(model)

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_points,
            num_restarts=20,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )

        new_y = self.problem.evaluate(candidates).to(device=self.device, dtype=self.dtype)

        self.train_x = torch.cat([self.train_x, candidates])
        self.train_y = torch.cat([self.train_y, new_y])

        return candidates, new_y