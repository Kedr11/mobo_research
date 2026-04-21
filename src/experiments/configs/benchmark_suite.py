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


def get_experiment_config(suite_name: str = "paper_suite"):
    suite = BENCHMARK_SUITES[suite_name]
    return {
        "suite_name": suite_name,
        "methods": list(suite["methods"]),
        "benchmark_names": list(suite["benchmark_names"]),
        "n_seeds": suite["n_seeds"],
        "primary_metric": suite["primary_metric"],
        "secondary_metrics": list(suite["secondary_metrics"]),
        "dtype": torch.float64,
    }


DEFAULT_EXPERIMENT_CONFIG = get_experiment_config("paper_suite")
