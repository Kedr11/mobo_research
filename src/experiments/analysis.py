from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _to_finite_float(value):
    if value is None or value == "":
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def get_pareto_front(y: torch.Tensor):
    if y.shape[0] <= 1:
        return y

    is_efficient = torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
    for i, target in enumerate(y):
        if not is_efficient[i]:
            continue

        dominators = torch.all(y >= target, dim=1) & torch.any(y > target, dim=1)
        if torch.any(dominators):
            is_efficient[i] = False
            continue

        dominated_by_target = torch.all(y <= target, dim=1)
        is_efficient[dominated_by_target] = False
        is_efficient[i] = True

    return y[is_efficient]


def _partition_hv_space(points: torch.Tensor):
    dim = points.shape[1]
    if len(points) == 0:
        return []

    if dim == 1:
        zero = torch.tensor([0.0], dtype=points.dtype, device=points.device)
        return [(zero, torch.max(points, dim=0)[0])]

    sorted_idx = torch.argsort(points[:, -1])
    points = points[sorted_idx]

    boxes = []
    for i in range(len(points)):
        lower_dim_boxes = _partition_hv_space(points[i + 1 :, :-1])
        h_low = points[i - 1, -1] if i > 0 else torch.tensor(0.0, dtype=points.dtype, device=points.device)
        h_high = points[i, -1]

        if h_high > h_low:
            for lower_box, upper_box in lower_dim_boxes:
                new_lower = torch.cat([lower_box, h_low.unsqueeze(0)])
                new_upper = torch.cat([upper_box, h_high.unsqueeze(0)])
                boxes.append((new_lower, new_upper))
    return boxes


def calculate_hypervolume(pareto_y: torch.Tensor, ref_point: torch.Tensor):
    shifted_points = pareto_y - ref_point
    valid_points = shifted_points[torch.all(shifted_points > 0, dim=1)]
    boxes = _partition_hv_space(valid_points)
    volume = 0.0
    for lower, upper in boxes:
        volume += torch.prod(upper - lower).item()
    return volume


def invert_to_minimization(y_max: torch.Tensor) -> torch.Tensor:
    return -y_max


def get_observed_pareto_front_minimization(observed_y_max: torch.Tensor) -> torch.Tensor:
    pareto_y_max = get_pareto_front(observed_y_max.detach().cpu())
    return invert_to_minimization(pareto_y_max)


def compute_hypervolume_minimization(pareto_y_min: torch.Tensor, ref_point_min: torch.Tensor) -> float:
    pareto_y_max = -pareto_y_min
    ref_point_max = -ref_point_min
    return float(calculate_hypervolume(pareto_y_max, ref_point_max))


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def compute_gd(pareto_y_min: np.ndarray, true_pf: np.ndarray) -> float:
    if len(pareto_y_min) == 0 or len(true_pf) == 0:
        return float("nan")
    distances = pairwise_distances(pareto_y_min, true_pf)
    return float(distances.min(axis=1).mean())


def compute_igd(pareto_y_min: np.ndarray, true_pf: np.ndarray) -> float:
    if len(pareto_y_min) == 0 or len(true_pf) == 0:
        return float("nan")
    distances = pairwise_distances(true_pf, pareto_y_min)
    return float(distances.min(axis=1).mean())


def compute_spacing(pareto_y_min: np.ndarray) -> float:
    if len(pareto_y_min) <= 2:
        return 0.0
    distances = pairwise_distances(pareto_y_min, pareto_y_min)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(nearest.std())


def compute_coverage(pareto_y_min: np.ndarray, true_pf: np.ndarray) -> float:
    if len(pareto_y_min) == 0 or len(true_pf) == 0:
        return float("nan")

    observed_range = np.maximum(pareto_y_min.max(axis=0) - pareto_y_min.min(axis=0), 0.0)
    true_range = np.maximum(true_pf.max(axis=0) - true_pf.min(axis=0), 1e-8)
    coverage = np.clip(observed_range / true_range, 0.0, 1.0)
    return float(np.mean(coverage))


def collect_iteration_metrics(problem, observed_y_max: torch.Tensor, iteration: int, evaluations: int):
    observed_y_max = observed_y_max.detach().cpu()
    observed_y_min = invert_to_minimization(observed_y_max)
    pareto_y_min = get_observed_pareto_front_minimization(observed_y_max)
    true_pf = problem.get_true_pareto_front()
    ref_point_min = problem.get_minimization_ref_point().detach().cpu()

    hypervolume = compute_hypervolume_minimization(pareto_y_min, ref_point_min)
    ideal_hypervolume = None
    normalized_hypervolume = None
    gd = float("nan")
    igd = float("nan")
    spacing = float("nan")
    coverage = float("nan")

    if true_pf is not None and len(true_pf) > 0:
        if getattr(problem, "_ideal_hypervolume", None) is None:
            true_pf_tensor = torch.tensor(true_pf, dtype=torch.float64)
            problem._ideal_hypervolume = compute_hypervolume_minimization(true_pf_tensor, ref_point_min)
        ideal_hypervolume = problem._ideal_hypervolume
        normalized_hypervolume = hypervolume / ideal_hypervolume if ideal_hypervolume > 0 else float("nan")
        pareto_np = pareto_y_min.numpy()
        gd = compute_gd(pareto_np, true_pf)
        igd = compute_igd(pareto_np, true_pf)
        spacing = compute_spacing(pareto_np)
        coverage = compute_coverage(pareto_np, true_pf)

    best_objectives = observed_y_min.min(dim=0).values.numpy()
    metrics = {
        "iteration": iteration,
        "evaluations": evaluations,
        "hypervolume": hypervolume,
        "ideal_hypervolume": ideal_hypervolume,
        "normalized_hypervolume": normalized_hypervolume,
        "gd": gd,
        "igd": igd,
        "spacing": spacing,
        "coverage": coverage,
        "pareto_size": int(pareto_y_min.shape[0]),
    }

    for idx, value in enumerate(best_objectives, start=1):
        metrics[f"best_f{idx}"] = float(value)

    return metrics, pareto_y_min.numpy(), observed_y_min.numpy()


def _prepare_output_path(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_pareto_scatter(problem, observed_y_min: np.ndarray, pareto_y_min: np.ndarray, output_path, method: str, seed: int, n_init: int = None):
    output_path = _prepare_output_path(output_path)
    true_pf = problem.get_true_pareto_front()

    initial_y = observed_y_min[:n_init] if n_init is not None else None
    acquired_y = observed_y_min[n_init:] if n_init is not None else observed_y_min

    if problem.num_objectives == 2:
        plt.figure(figsize=(7, 5))
        if true_pf is not None:
            plt.plot(true_pf[:, 0], true_pf[:, 1], color="black", alpha=0.45, label="True PF")

        if initial_y is not None and len(initial_y) > 0:
            plt.scatter(initial_y[:, 0], initial_y[:, 1], s=34, alpha=0.7, color="#6EDFF6", label="Initial")
        if acquired_y is not None and len(acquired_y) > 0:
            plt.scatter(acquired_y[:, 0], acquired_y[:, 1], s=30, alpha=0.78, color="#2F6BFF", label="Acquired")
        plt.scatter(
            pareto_y_min[:, 0],
            pareto_y_min[:, 1],
            s=36,
            alpha=0.95,
            color="#E45756",
            edgecolors="black",
            linewidths=0.3,
            label="Pareto",
        )
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title(f"{problem.name.upper()} | {method} | seed={seed}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    if problem.num_objectives == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if true_pf is not None:
            ax.scatter(true_pf[:, 0], true_pf[:, 1], true_pf[:, 2], s=8, alpha=0.12, color="black", label="True PF")
        if initial_y is not None and len(initial_y) > 0:
            ax.scatter(initial_y[:, 0], initial_y[:, 1], initial_y[:, 2], s=34, alpha=0.7, color="#6EDFF6", label="Initial")
        if acquired_y is not None and len(acquired_y) > 0:
            ax.scatter(acquired_y[:, 0], acquired_y[:, 1], acquired_y[:, 2], s=30, alpha=0.78, color="#2F6BFF", label="Acquired")
        ax.scatter(
            pareto_y_min[:, 0],
            pareto_y_min[:, 1],
            pareto_y_min[:, 2],
            s=36,
            alpha=0.95,
            color="#E45756",
            edgecolors="black",
            linewidths=0.3,
            label="Pareto",
        )
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.set_title(f"{problem.name.upper()} | {method} | seed={seed}")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    plt.figure(figsize=(7, 5))
    plt.text(0.5, 0.5, f"{problem.num_objectives}D Pareto scatter is not implemented", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_metric_convergence_plot(iteration_rows, output_path, problem_name: str = None, metric_name: str = "normalized_hypervolume"):
    output_path = _prepare_output_path(output_path)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in iteration_rows:
        if problem_name is not None and row["problem_name"] != problem_name:
            continue
        metric_value = _to_finite_float(row.get(metric_name))
        if metric_value is None:
            continue
        grouped[row["method"]][int(row["iteration"])].append(metric_value)

    if not grouped:
        return

    plt.figure(figsize=(8, 5))
    for method, iter_map in grouped.items():
        iterations = sorted(iter_map)
        means = np.array([np.mean(iter_map[i]) for i in iterations], dtype=float)
        stds = np.array([np.std(iter_map[i]) for i in iterations], dtype=float)
        plt.plot(iterations, means, linewidth=2, label=method)
        plt.fill_between(iterations, means - stds, means + stds, alpha=0.18)

    plt.xlabel("Iteration")
    ylabel = "Normalized Hypervolume" if metric_name == "normalized_hypervolume" else metric_name.replace("_", " ").title()
    plt.ylabel(ylabel)
    title_prefix = problem_name.upper() if problem_name is not None else "All benchmarks"
    plt.title(f"{title_prefix} | {ylabel} convergence")
    if metric_name == "normalized_hypervolume":
        plt.ylim(bottom=0.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_metric_boxplot(summary_rows, output_path, metric_name: str, title: str, problem_name: str = None):
    output_path = _prepare_output_path(output_path)
    grouped = defaultdict(list)
    for row in summary_rows:
        if problem_name is not None and row["problem_name"] != problem_name:
            continue
        metric_value = _to_finite_float(row.get(metric_name))
        if metric_value is None:
            continue
        grouped[row["method"]].append(metric_value)

    if not grouped:
        return

    labels = list(grouped.keys())
    values = [grouped[label] for label in labels]
    positions = np.arange(1, len(labels) + 1)

    plt.figure(figsize=(8, 5))
    if any(len(series) >= 2 for series in values):
        boxplot = plt.boxplot(values, labels=labels, patch_artist=True)
        fill_colors = ["#7DB7FF", "#FFB36E", "#8FD694", "#F28E8E"]
        for patch, color in zip(boxplot["boxes"], fill_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

    for position, series in zip(positions, values):
        jitter = np.linspace(-0.04, 0.04, len(series)) if len(series) > 1 else np.array([0.0])
        plt.scatter(
            np.full(len(series), position) + jitter,
            series,
            s=46,
            color="#2F6BFF",
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
            zorder=3,
        )

    plt.xticks(positions, labels)
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
