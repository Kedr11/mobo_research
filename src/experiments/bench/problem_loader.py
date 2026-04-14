import torch
import numpy as np
from pymoo.problems import get_problem
from torch import Tensor


class BenchmarkProblem:
    """
    Класс-адаптер: берет задачу из Pymoo и готовит её для BoTorch.
    """

    def __init__(self, name: str = "dtlz2", n_var: int = 6, n_obj: int = 2):
        self.pymoo_problem = get_problem(name, n_var=n_var, n_obj=n_obj)

        self.name = name
        self.dim = self.pymoo_problem.n_var
        self.num_objectives = self.pymoo_problem.n_obj

        xl, xu = self.pymoo_problem.bounds()
        self.bounds = torch.tensor(
            np.stack([xl, xu]),
            dtype=torch.float64
        )

    def evaluate(self, x: Tensor) -> Tensor:
        """
        Принимает тензор параметров от BoTorch, считает результат через Pymoo
        и возвращает тензор целей.
        """
        # Переводим тензор в numpy для Pymoo
        x_np = x.detach().cpu().numpy()

        # Pymoo ожидает 2D массив [количество_точек, размерность]
        if x_np.ndim == 1:
            x_np = x_np[None, :]

        # Считаем значение функции
        f_values = self.pymoo_problem.evaluate(x_np)

        # Чтобы BoTorch работал корректно, мы возвращаем значения с минусом.
        return torch.tensor(-f_values, dtype=torch.float64)

    def get_ref_point(self):
        """
        Возвращает точку отсчета (Reference Point) для расчета гиперобъема.
        """
        # Пример для DTLZ2 с 2 целями (из твоей ссылки):
        if self.name == "dtlz2" and self.num_objectives == 2:
            # В Optuna Hub точка указана для минимизации (например, 1.1, 1.1).
            # Так как мы максимизируем (-f), точка превращается в -1.1
            return torch.tensor([-1.1, -1.1], dtype=torch.float64)

        # Заглушка для других задач (нужно будет дополнить по мере выбора задач)
        return torch.tensor([-2.0] * self.num_objectives, dtype=torch.float64)