from copy import deepcopy

import numpy as np
import torch
from pymoo.problems import get_problem
from torch import Tensor

from tests.benchmark_suite_data import BENCHMARK_SUITES, BENCHMARK_TASKS


def get_benchmark_config(name: str):
    normalized_name = name.lower()
    if normalized_name not in BENCHMARK_TASKS:
        available = ", ".join(sorted(BENCHMARK_TASKS))
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
    return deepcopy(BENCHMARK_TASKS[normalized_name])


def list_benchmark_configs():
    return {name: deepcopy(config) for name, config in BENCHMARK_TASKS.items()}


def get_suite_config(name: str):
    if name not in BENCHMARK_SUITES:
        available = ", ".join(sorted(BENCHMARK_SUITES))
        raise KeyError(f"Unknown suite '{name}'. Available: {available}")
    return deepcopy(BENCHMARK_SUITES[name])


class BenchmarkProblem:
    def __init__(self, name: str = "dtlz2", n_var: int = None, n_obj: int = None, config: dict = None):
        benchmark_config = get_benchmark_config(name)
        if config is not None:
            benchmark_config.update(config)

        self.config = benchmark_config
        self.name = benchmark_config["problem_name"].lower()
        self.description = benchmark_config.get("description", "")
        self.difficulty = benchmark_config.get("difficulty", "unknown")
        self.family = benchmark_config.get("family", "")
        self.tags = list(benchmark_config.get("tags", []))
        self.pf_points = int(benchmark_config.get("pf_points", 200))

        resolved_n_var = int(n_var or benchmark_config["n_var"])
        resolved_n_obj = int(n_obj or benchmark_config["n_obj"])

        if self.name.startswith("zdt"):
            self.pymoo_problem = get_problem(self.name, n_var=resolved_n_var)
        else:
            self.pymoo_problem = get_problem(self.name, n_var=resolved_n_var, n_obj=resolved_n_obj)

        self.dim = self.pymoo_problem.n_var
        self.num_objectives = self.pymoo_problem.n_obj

        xl, xu = self.pymoo_problem.bounds()
        self.bounds = torch.tensor(np.stack([xl, xu]), dtype=torch.float64)
        self._true_pareto_front = None
        self._min_ref_point = None
        self._max_ref_point = None

    def evaluate(self, x: Tensor) -> Tensor:
        lower = self.bounds[0].to(device=x.device, dtype=x.dtype)
        upper = self.bounds[1].to(device=x.device, dtype=x.dtype)
        x_clamped = torch.max(torch.min(x, upper), lower)
        x_np = x_clamped.detach().cpu().numpy()

        if x_np.ndim == 1:
            x_np = x_np[None, :]

        f_values = self.pymoo_problem.evaluate(x_np)

        if np.isnan(f_values).any():
            print(f"WARNING: Pymoo returned NaN for {self.name}. Input X: {x_np}")

        return torch.tensor(-f_values, device=x.device, dtype=torch.float64)

    def get_true_pareto_front(self):
        if self._true_pareto_front is None:
            try:
                pf = self.pymoo_problem.pareto_front(n_pareto_points=self.pf_points)
            except TypeError:
                pf = self.pymoo_problem.pareto_front()

            if pf is None:
                return None
            self._true_pareto_front = np.asarray(pf, dtype=np.float64)

        return self._true_pareto_front.copy()

    def get_minimization_ref_point(self):
        if self._min_ref_point is None:
            pf = self.get_true_pareto_front()
            if pf is None:
                self._min_ref_point = torch.ones(self.num_objectives, dtype=torch.float64)
            else:
                pf_min = pf.min(axis=0)
                pf_max = pf.max(axis=0)
                span = np.maximum(pf_max - pf_min, 1e-3)
                self._min_ref_point = torch.tensor(pf_max + 0.1 * span, dtype=torch.float64)
        return self._min_ref_point.clone()

    def get_ref_point(self):
        if self._max_ref_point is None:
            self._max_ref_point = -self.get_minimization_ref_point()
        return self._max_ref_point.clone().to(device=self.bounds.device, dtype=torch.float64)

    def to_metadata(self):
        return {
            "problem_name": self.name,
            "family": self.family,
            "description": self.description,
            "difficulty": self.difficulty,
            "n_var": self.dim,
            "n_obj": self.num_objectives,
            "pf_points": self.pf_points,
            "tags": ",".join(self.tags),
        }
