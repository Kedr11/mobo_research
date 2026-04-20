import csv
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.experiments.analysis import (
    collect_iteration_metrics,
    save_metric_boxplot,
    save_metric_convergence_plot,
    save_pareto_scatter,
)
from src.experiments.bench.problem_loader import BenchmarkProblem, get_benchmark_config
from src.experiments.configs.benchmark_suite import DEFAULT_EXPERIMENT_CONFIG
from src.experiments.runners.botorch_runner import BotorchRunner


EXPERIMENT_CONFIG = {
    **DEFAULT_EXPERIMENT_CONFIG,
    "output_root": "outputs",
}


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2, default=str)


def build_runner(problem, method, device, dtype):
    if method not in {"ParEGO", "qEHVI"}:
        raise ValueError(f"Unsupported method '{method}'.")
    return BotorchRunner(problem, acq_type=method, device=device, dtype=dtype)


def flatten_runner_info(runner_info):
    flat = {}
    for key, value in runner_info.items():
        if isinstance(value, list):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    return flat


def run_single_experiment(problem_config, method, seed, device, dtype, suite_figures_dir):
    set_global_seed(seed)
    problem = BenchmarkProblem(config=problem_config, name=problem_config["problem_name"])
    runner = build_runner(problem, method, device, dtype)
    runner.initialize_data(n=problem_config["n_init"])

    iteration_rows = []
    start_time = perf_counter()

    initial_metrics, pareto_y_min, observed_y_min = collect_iteration_metrics(
        problem=problem,
        observed_y_max=runner.train_y,
        iteration=0,
        evaluations=runner.train_y.shape[0],
    )
    iteration_rows.append(
        {
            "problem_name": problem.name,
            "method": method,
            "seed": seed,
            "acq_type": method,
            **initial_metrics,
        }
    )

    for iteration in range(1, problem_config["n_iter"] + 1):
        runner.run_iteration(n_points=1)
        raw_runner_info = flatten_runner_info(runner.last_iteration_info)
        metrics, pareto_y_min, observed_y_min = collect_iteration_metrics(
            problem=problem,
            observed_y_max=runner.train_y,
            iteration=iteration,
            evaluations=runner.train_y.shape[0],
        )
        iteration_rows.append(
            {
                "problem_name": problem.name,
                "method": method,
                "seed": seed,
                **raw_runner_info,
                **metrics,
            }
        )

    elapsed_sec = perf_counter() - start_time
    final_row = dict(iteration_rows[-1])
    final_row["elapsed_sec"] = elapsed_sec
    final_row["difficulty"] = problem.difficulty
    final_row["family"] = problem.family
    final_row["description"] = problem.description
    final_row["n_var"] = problem.dim
    final_row["n_obj"] = problem.num_objectives

    save_pareto_scatter(
        problem=problem,
        observed_y_min=observed_y_min,
        pareto_y_min=pareto_y_min,
        output_path=suite_figures_dir / problem.name / f"{method}_seed_{seed:02d}_pareto.png",
        method=method,
        seed=seed,
        n_init=problem_config["n_init"],
    )

    pareto_rows = [
        {
            "problem_name": problem.name,
            "method": method,
            "seed": seed,
            **{f"f{i + 1}": float(values[i]) for i in range(problem.num_objectives)},
        }
        for values in pareto_y_min
    ]

    return iteration_rows, final_row, pareto_rows, problem


def save_problem_level_figures(iteration_rows, summary_rows, problem):
    output_root = Path(EXPERIMENT_CONFIG["output_root"])
    suite_name = EXPERIMENT_CONFIG["suite_name"]
    figures_dir = ensure_dir(output_root / "figures" / suite_name / problem.name)

    save_metric_convergence_plot(
        iteration_rows=iteration_rows,
        output_path=figures_dir / "hypervolume_convergence.png",
        problem_name=problem.name,
        metric_name="normalized_hypervolume",
    )

    save_metric_boxplot(
        summary_rows=summary_rows,
        output_path=figures_dir / "final_normalized_hv_boxplot.png",
        metric_name="normalized_hypervolume",
        title=f"{problem.name.upper()} | Final normalized hypervolume",
        problem_name=problem.name,
    )


def save_global_figures(summary_rows):
    output_root = Path(EXPERIMENT_CONFIG["output_root"])
    suite_name = EXPERIMENT_CONFIG["suite_name"]
    figures_dir = ensure_dir(output_root / "figures" / suite_name / "global")

    save_metric_boxplot(
        summary_rows=summary_rows,
        output_path=figures_dir / "global_normalized_hv_boxplot.png",
        metric_name="normalized_hypervolume",
        title="All benchmarks | Final normalized hypervolume",
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    output_root = ensure_dir(EXPERIMENT_CONFIG["output_root"])
    suite_name = EXPERIMENT_CONFIG["suite_name"]
    suite_logs_dir = ensure_dir(output_root / "logs" / suite_name)
    suite_figures_dir = ensure_dir(output_root / "figures" / suite_name)

    print(f"Running suite '{suite_name}' on {device}.")

    benchmark_rows = []
    all_iteration_rows = []
    all_summary_rows = []
    all_pareto_rows = []
    problems_by_name = {}

    for benchmark_name in EXPERIMENT_CONFIG["benchmark_names"]:
        problem_config = get_benchmark_config(benchmark_name)
        benchmark_rows.append(
            {
                **problem_config,
                "tags": ",".join(problem_config.get("tags", [])),
            }
        )

        for method in EXPERIMENT_CONFIG["methods"]:
            for seed in range(EXPERIMENT_CONFIG["n_seeds"]):
                print(f"[{benchmark_name.upper()}] method={method} seed={seed}")
                iteration_rows, summary_row, pareto_rows, problem = run_single_experiment(
                    problem_config=problem_config,
                    method=method,
                    seed=seed,
                    device=device,
                    dtype=EXPERIMENT_CONFIG["dtype"],
                    suite_figures_dir=suite_figures_dir,
                )
                problems_by_name[problem.name] = problem
                all_iteration_rows.extend(iteration_rows)
                all_summary_rows.append(summary_row)
                all_pareto_rows.extend(pareto_rows)

    for problem_name in EXPERIMENT_CONFIG["benchmark_names"]:
        save_problem_level_figures(
            iteration_rows=all_iteration_rows,
            summary_rows=all_summary_rows,
            problem=problems_by_name[problem_name],
        )

    save_global_figures(all_summary_rows)

    write_csv(all_iteration_rows, suite_logs_dir / "iteration_metrics.csv")
    write_csv(all_summary_rows, suite_logs_dir / "run_summary.csv")
    write_csv(all_pareto_rows, suite_logs_dir / "final_pareto_points.csv")
    write_csv(benchmark_rows, suite_logs_dir / "benchmark_catalog.csv")
    save_json(EXPERIMENT_CONFIG, suite_logs_dir / "experiment_config.json")

    print(f"Logs saved to {suite_logs_dir}")
    print(f"Figures saved to {suite_figures_dir}")


if __name__ == "__main__":
    main()
