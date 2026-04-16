import torch
import numpy as np
from src.experiments.bench.problem_loader import BenchmarkProblem
from src.experiments.runners.botorch_runner import BotorchRunner
from pymoo.visualization.scatter import Scatter

# --- ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
CONFIG = {
    "problem_name": "dtlz2",  # Попробуй "zdt1" (2 цели) или "dtlz2" (3 цели)
    "n_var": 6,  # Количество переменных (для zdt1 обычно 30, но начни с 6)
    "n_obj": 3,  # Количество целевых функций
    "n_init": 40,  # Начальные точки (для GPU можно смело 80+)
    "n_iter": 50,  # Количество итераций оптимизации
    "acq_type": "qEHVI",  # Стратегия: "qEHVI" или "ParEGO"
    "dtype": torch.float32,  # float64 намного быстрее на RTX 3050 (Tensor Cores)
}


def main():
    # 1. Настройка GPU (RTX 3050)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision('high')
    print(f"🚀 Running on: {device} | Problem: {CONFIG['problem_name'].upper()}")

    # 2. Инициализация задачи и оптимизатора
    # ZDT всегда имеет 2 цели, поэтому n_obj там указывать не нужно
    if CONFIG["problem_name"].startswith("zdt"):
        actual_n_obj = 2
        problem = BenchmarkProblem(name=CONFIG["problem_name"], n_var=CONFIG["n_var"])
    else:
        actual_n_obj = CONFIG["n_obj"]
        problem = BenchmarkProblem(name=CONFIG["problem_name"], n_var=CONFIG["n_var"], n_obj=actual_n_obj)

    runner = BotorchRunner(
        problem,
        acq_type=CONFIG["acq_type"],
        device=device,
        dtype=CONFIG["dtype"]
    )

    # 3. Разведка (Initial Sampling)
    print(f"📊 Initializing with {CONFIG['n_init']} points...")
    runner.initialize_data(n=CONFIG["n_init"])

    # 4. Цикл оптимизации
    print(f"⚡ Starting {CONFIG['n_iter']} MOBO iterations...")
    for i in range(CONFIG["n_iter"]):
        candidates, new_y = runner.run_iteration(n_points=1)

        # Диагностика: выводим Y и X
        y_vals = [round(val, 4) for val in new_y.squeeze().tolist()]
        x_vals = [round(val, 3) for val in candidates.squeeze().tolist()]

        # Считаем среднее отклонение переменных g(x) от 0.5 (для DTLZ/ZDT)
        g_dist = np.mean([abs(x - 0.5) for x in x_vals[problem.num_objectives - 1:]])

        print(f"Iter {i + 1}/{CONFIG['n_iter']} | Y: {y_vals} | g-dist: {g_dist:.3f}")

    # 5. Визуализация
    # Инвертируем обратно (т.к. BoTorch максимизирует -f)
    train_y_np = -runner.train_y.detach().cpu().numpy()
    pf = problem.pymoo_problem.pareto_front()

    title = f"MOBO: {CONFIG['problem_name']} ({actual_n_obj} Obj)"
    labels = [f"f{i + 1}" for i in range(actual_n_obj)]

    plot = Scatter(title=title, labels=labels)

    if pf is not None:
        # КРИТИЧНО: для 3D (DTLZ2) убираем plot_type="line",
        # так как в 3D pymoo не умеет рисовать "линию" через сетку
        plot.add(pf, plot_type="line", color="black", alpha=0.3, label="True PF")
        if actual_n_obj <= 2:
            plot.add(pf, plot_type="line", color="black", alpha=0.3, label="True PF")

    plot.add(train_y_np[:CONFIG['n_init']], color="blue", alpha=0.3, label="Initial")
    plot.add(train_y_np[CONFIG['n_init']:], color="red", s=30, label="Optimized")
    plot.show()

if __name__ == "__main__":
    main()