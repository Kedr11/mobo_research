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


class BotorchRunner:
    def __init__(self, problem, acq_type="qEHVI", device=None, dtype=torch.float64):
        """
        Инициализация 'двигателя' оптимизации.
        :param problem: наш объект BenchmarkProblem (лоадер задач)
        :param acq_type: стратегия поиска (qEHVI — объем, ParEGO — веса)
        """
        self.problem = problem
        self.acq_type = acq_type
        self.device = device if device else torch.device("cpu")
        self.dtype = dtype

        # Списки тензоров, где мы будем хранить всю историю: что ввели (X) и что получили (Y)
        self.train_x = torch.empty(0, problem.dim, device=self.device, dtype=self.dtype)
        self.train_y = torch.empty(0, problem.num_objectives, device=self.device, dtype=self.dtype)

        # Границы задачи, переведенные на нужный девайс (CPU/GPU)
        self.bounds = problem.bounds.to(device=self.device, dtype=self.dtype)

    def initialize_data(self, n=10):
        """
        ШАГ 1: Разведка.
        Мы не можем запустить ИИ на пустом месте, ему нужны начальные данные.
        Используем метод Соболя для равномерного покрытия пространства.
        """
        from botorch.utils.sampling import draw_sobol_samples

        # Рисуем 'сетку' начальных точек
        new_x = draw_sobol_samples(bounds=self.bounds, n=n, q=1).squeeze(1)
        # Считаем реальные значения функций (через наш мостик в Pymoo)
        new_y = self.problem.evaluate(new_x).to(device=self.device, dtype=self.dtype)

        # Добавляем в 'память' класса
        self.train_x = torch.cat([self.train_x, new_x])
        self.train_y = torch.cat([self.train_y, new_y])

    def _get_fitted_model(self):
        """
        ШАГ 2: Построение суррогата (модели мира).
        Мы строим по одной Гауссовской модели на каждый критерий.
        """
        models = []
        for i in range(self.problem.num_objectives):
            train_y_slice = self.train_y[:, i: i + 1]  # Берем данные только по 1 критерию

            # Обучаем модель с автоматической нормализацией внутри
            gp = SingleTaskGP(
                train_X=self.train_x,
                train_Y=train_y_slice,
                input_transform=Normalize(d=self.problem.dim, bounds=self.bounds),
                outcome_transform=Standardize(m=1)
            )
            models.append(gp)

        # Объединяем их в ансамбль (ModelList)
        model_list = ModelListGP(*models)
        # Находим такие параметры модели, чтобы она лучше всего описывала наши данные
        mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
        fit_gpytorch_mll(mll)
        return model_list

    def _get_acq_func(self, model):
        """
        ШАГ 3: Стратегия выбора.
        Решаем, как именно мы будем искать следующую точку.
        """
        # Самплер Соболя помогает 'предсказывать будущее' точнее и быстрее
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        if self.acq_type == "qEHVI":
            # Тянем Reference Point из загрузчика (твои 1.1, 1.1 превращенные в -1.1)
            ref_point = self.problem.get_ref_point().to(device=self.device, dtype=self.dtype)
            return qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                sampler=sampler,
                partitioning_strategy="fast-generic"
            )

        elif self.acq_type == "ParEGO":
            # Генерируем случайные веса, чтобы смешать критерии в один коктейль
            weights = torch.randn(self.problem.num_objectives, device=self.device, dtype=self.dtype).abs()
            weights /= weights.sum()  # Сумма весов всегда = 1

            # Показываем модели, как складывать её выходы
            post_transform = ScalarizedPosteriorTransform(weights=weights)

            # Находим лучшее значение среди уже скаляризованных для базы qEI
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
        Самый главный метод, который делает один шаг оптимизации.
        """
        # 1. Спрашиваем у суррогата: 'Что ты думаешь о задаче сейчас?'
        model = self._get_fitted_model()

        # 2. Спрашиваем у стратегии: 'Какая область самая перспективная?'
        acq_func = self._get_acq_func(model)

        # 3. Ищем максимум перспективности (L-BFGS-B 'бежит в гору')
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=n_points,
            num_restarts=20,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )

        # 4. Проверяем теорию практикой: считаем реальное значение в DTLZ
        new_y = self.problem.evaluate(candidates).to(device=self.device, dtype=self.dtype)

        # 5. Сохраняем опыт
        self.train_x = torch.cat([self.train_x, candidates])
        self.train_y = torch.cat([self.train_y, new_y])

        return candidates, new_y