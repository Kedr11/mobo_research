import torch

from tests.benchmark_suite_data import BENCHMARK_SUITES, BENCHMARK_TASKS


BENCHMARK_RUN_CONFIGS = {
    name: {
        "problem_name": config["problem_name"],
        "n_var": config["n_var"],
        "n_obj": config["n_obj"],
        "n_init": config["n_init"],
        "n_iter": config["n_iter"],
        "difficulty": config["difficulty"],
        "description": config["description"],
        "pf_points": config["pf_points"],
        "tags": list(config["tags"]),
    }
    for name, config in BENCHMARK_TASKS.items()
}


DEFAULT_EXPERIMENT_CONFIG = {
    "suite_name": "paper_suite",
    "methods": list(BENCHMARK_SUITES["paper_suite"]["methods"]),
    "benchmark_names": list(BENCHMARK_SUITES["paper_suite"]["benchmark_names"]),
    "n_seeds": BENCHMARK_SUITES["paper_suite"]["n_seeds"],
    "primary_metric": BENCHMARK_SUITES["paper_suite"]["primary_metric"],
    "secondary_metrics": list(BENCHMARK_SUITES["paper_suite"]["secondary_metrics"]),
    "dtype": torch.float64,
}
