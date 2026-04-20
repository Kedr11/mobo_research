import csv
import gc
import json
import random
import sys
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
    "method_devices": {
        "ParEGO": "cpu",
        "qEHVI": "cpu",
    },
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


def append_csv_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    if file_exists:
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = list(reader.fieldnames or [])
    else:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(input_path):
    input_path = Path(input_path)
    if not input_path.exists() or input_path.stat().st_size == 0:
        return []
    with input_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2, default=str)


def resolve_runner_device(method, default_device):
    requested_device = EXPERIMENT_CONFIG.get("method_devices", {}).get(method, "cpu")
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print(
            f"{method} fallback: CUDA requested, but unavailable in the current PyTorch build. Switching to CPU.",
            flush=True,
        )
        return torch.device("cpu")

    if requested_device == "auto":
        return default_device

    return torch.device("cpu")


def build_runner(problem, method, device, dtype):
    if method not in {"ParEGO", "qEHVI"}:
        raise ValueError(f"Unsupported method '{method}'.")

    runner_device = resolve_runner_device(method, default_device=device)

    return BotorchRunner(
        problem,
        acq_type=method,
        device=runner_device,
        dtype=dtype,
        verbose=False,
        keep_history=False,
    )


def cleanup_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    runner_device = runner.device
    print(
        f"[{problem.name.upper()} | {method} | seed={seed}] "
        f"start on {runner_device.type} | n_init={problem_config['n_init']} | n_iter={problem_config['n_iter']}",
        flush=True,
    )
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
        print(
            f"[{problem.name.upper()} | {method} | seed={seed}] "
            f"iteration {iteration}/{problem_config['n_iter']}",
            flush=True,
        )
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
    final_row["device"] = runner_device.type
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

    print(
        f"[{problem.name.upper()} | {method} | seed={seed}] "
        f"done in {elapsed_sec:.1f}s | final normalized HV={final_row.get('normalized_hypervolume', float('nan')):.4f}",
        flush=True,
    )

    return iteration_rows, final_row, pareto_rows, problem, runner_device


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

    iteration_rows = read_csv_rows(output_root / "logs" / suite_name / "iteration_metrics.csv")
    save_metric_convergence_plot(
        iteration_rows=iteration_rows,
        output_path=figures_dir / "global_normalized_hv_convergence.png",
        metric_name="normalized_hypervolume",
    )

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
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_root = ensure_dir(EXPERIMENT_CONFIG["output_root"])
    suite_name = EXPERIMENT_CONFIG["suite_name"]
    suite_logs_dir = ensure_dir(output_root / "logs" / suite_name)
    suite_figures_dir = ensure_dir(output_root / "figures" / suite_name)
    iteration_csv_path = suite_logs_dir / "iteration_metrics.csv"
    summary_csv_path = suite_logs_dir / "run_summary.csv"
    pareto_csv_path = suite_logs_dir / "final_pareto_points.csv"
    benchmark_csv_path = suite_logs_dir / "benchmark_catalog.csv"
    config_json_path = suite_logs_dir / "experiment_config.json"

    print(f"Running suite '{suite_name}' with base device={device}.")
    print(f"Python: {sys.executable}")
    print(
        f"Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | "
        f"CUDA version: {torch.version.cuda} | device_count: {torch.cuda.device_count()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"GPU[0]: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Method devices: {EXPERIMENT_CONFIG.get('method_devices', {})}", flush=True)

    benchmark_rows = []
    problem_configs_by_name = {}
    total_runs = (
        len(EXPERIMENT_CONFIG["benchmark_names"])
        * len(EXPERIMENT_CONFIG["methods"])
        * EXPERIMENT_CONFIG["n_seeds"]
    )
    run_index = 0

    for benchmark_name in EXPERIMENT_CONFIG["benchmark_names"]:
        problem_config = get_benchmark_config(benchmark_name)
        problem_configs_by_name[benchmark_name] = problem_config
        benchmark_rows.append(
            {
                **problem_config,
                "tags": ",".join(problem_config.get("tags", [])),
            }
        )

    write_csv(benchmark_rows, benchmark_csv_path)
    save_json(EXPERIMENT_CONFIG, config_json_path)

    all_iteration_rows = read_csv_rows(iteration_csv_path)
    all_summary_rows = read_csv_rows(summary_csv_path)
    completed_runs = {
        (row["problem_name"], row["method"], int(row["seed"]))
        for row in all_summary_rows
    }

    for benchmark_name in EXPERIMENT_CONFIG["benchmark_names"]:
        problem_config = problem_configs_by_name[benchmark_name]

        for method in EXPERIMENT_CONFIG["methods"]:
            for seed in range(EXPERIMENT_CONFIG["n_seeds"]):
                run_index += 1
                run_key = (problem_config["problem_name"], method, seed)
                if run_key in completed_runs:
                    print(
                        f"[run {run_index}/{total_runs}] "
                        f"[{benchmark_name.upper()}] method={method} seed={seed} already completed, skipping",
                        flush=True,
                    )
                    continue

                print(
                    f"[run {run_index}/{total_runs}] "
                    f"[{benchmark_name.upper()}] method={method} seed={seed}",
                    flush=True,
                )
                iteration_rows, summary_row, pareto_rows, problem, runner_device = run_single_experiment(
                    problem_config=problem_config,
                    method=method,
                    seed=seed,
                    device=device,
                    dtype=EXPERIMENT_CONFIG["dtype"],
                    suite_figures_dir=suite_figures_dir,
                )
                append_csv_rows(iteration_rows, iteration_csv_path)
                append_csv_rows([summary_row], summary_csv_path)
                append_csv_rows(pareto_rows, pareto_csv_path)

                all_iteration_rows.extend(iteration_rows)
                all_summary_rows.append(summary_row)
                completed_runs.add(run_key)

                save_problem_level_figures(
                    iteration_rows=all_iteration_rows,
                    summary_rows=all_summary_rows,
                    problem=problem,
                )
                save_global_figures(all_summary_rows)
                cleanup_memory(runner_device)

    for problem_name in EXPERIMENT_CONFIG["benchmark_names"]:
        problem = BenchmarkProblem(config=problem_configs_by_name[problem_name], name=problem_name)
        save_problem_level_figures(
            iteration_rows=all_iteration_rows,
            summary_rows=all_summary_rows,
            problem=problem,
        )

    save_global_figures(all_summary_rows)

    print(f"Logs saved to {suite_logs_dir}")
    print(f"Figures saved to {suite_figures_dir}")
    cleanup_memory(device)


if __name__ == "__main__":
    main()
