import torch
import numpy as np
from pymoo.problems import get_problem
from torch import Tensor

class BenchmarkProblem:
    def __init__(self, name: str = "dtlz2", n_var: int = 6, n_obj: int = 2):
        if name.lower().startswith("zdt"):
            self.pymoo_problem = get_problem(name.lower(), n_var=n_var)
        else:
            self.pymoo_problem = get_problem(name.lower(), n_var=n_var, n_obj=n_obj)

        self.name = name
        self.dim = self.pymoo_problem.n_var
        self.num_objectives = self.pymoo_problem.n_obj

        xl, xu = self.pymoo_problem.bounds()
        self.bounds = torch.tensor(
            np.stack([xl, xu]),
            dtype=torch.float64
        )

    def evaluate(self, x: Tensor) -> Tensor:
        x_clamped = torch.clamp(x, 0.0, 1.0)
        x_np = x_clamped.detach().cpu().numpy()

        if x_np.ndim == 1:
            x_np = x_np[None, :]

        f_values = self.pymoo_problem.evaluate(x_np)

        if np.isnan(f_values).any():
            print(f"WARNING: Pymoo returned NaN for {self.name}. Input X: {x_np}")

        # Инверсия уже происходит здесь! BoTorch получает данные для максимизации.
        return torch.tensor(-f_values, device=x.device, dtype=torch.float64)

    def get_ref_point(self):
        return torch.tensor([-1.1] * self.num_objectives, device=self.bounds.device, dtype=torch.float64)