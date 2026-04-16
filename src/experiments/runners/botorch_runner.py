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

from gpytorch.priors import GammaPrior
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.models.transforms import Standardize, Normalize
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
        models = []
        for i in range(self.problem.num_objectives):
            train_y_slice = self.train_y[:, i: i + 1]

            # Используем Matern 5/2 с жестким априорным распределением
            # Это заставляет модель сохранять высокую неопределенность в неизученных зонах
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.problem.dim,
                    lengthscale_prior=GammaPrior(3.0, 6.0),  # Не дает "залипнуть" в одной точке
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )

            gp = SingleTaskGP(
                train_X=self.train_x,
                train_Y=train_y_slice,
                # Обязательно используем встроенные трансформеры
                input_transform=Normalize(d=self.problem.dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
                covar_module=covar_module
            )
            models.append(gp)

        model_list = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
        fit_gpytorch_mll(mll)
        return model_list

    #from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
    # Удали импорт infer_reference_point, он нам больше не нужен

    def _get_acq_func(self, model):
        """
        ШАГ 3: Стратегия выбора с жесткой фильтрацией мусорных точек.
        """
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        if self.acq_type == "qEHVI":
            with torch.no_grad():
                # 1. Используем твой фиксированный ref_point (например, [-1.1, -1.1])
                # Это заставит алгоритм игнорировать всё, что хуже этих значений.
                ref_point = self.problem.get_ref_point().to(device=self.device, dtype=self.dtype)

                # 2. Отбрасываем NaN
                valid_y = self.train_y[~torch.isnan(self.train_y).any(dim=1)]

                # 3. Важный нюанс: для построения разбиения (partitioning)
                # мы должны использовать только те точки, которые ЛУЧШЕ референса.
                # Если все точки хуже - BoTorch выдаст ошибку, поэтому добавим фильтрацию.
                better_than_ref = (valid_y > ref_point).all(dim=1)
                if better_than_ref.any():
                    y_for_partitioning = valid_y[better_than_ref]
                else:
                    # Если нормальных точек еще нет, берем все, но EHVI будет мал
                    y_for_partitioning = valid_y

                partitioning = NondominatedPartitioning(
                    ref_point=ref_point,
                    Y=y_for_partitioning
                )

            return qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )

        elif self.acq_type == "ParEGO":
            # Код для ParEGO оставляем без изменений, он работает по другому принципу
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
            num_restarts=5,
            raw_samples=256,

        )
        # После создания модели
        posterior = model.posterior(candidates)
        print(f"Model Mean: {posterior.mean.detach().squeeze()}")
        print(f"Model Std: {posterior.variance.sqrt().detach().squeeze()}")
        new_y = self.problem.evaluate(candidates).to(device=self.device, dtype=self.dtype)

        self.train_x = torch.cat([self.train_x, candidates])
        self.train_y = torch.cat([self.train_y, new_y])

        return candidates, new_y