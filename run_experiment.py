import torch
from src.experiments.bench.problem_loader import BenchmarkProblem  # Убедись, что путь импорта совпадает с твоей структурой папок
from src.experiments.runners.botorch_runner import BotorchRunner
from pymoo.visualization.scatter import Scatter

def main():
    # 1. Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Создаем задачу: DTLZ2 с 3 целями
    problem = BenchmarkProblem(name="dtlz2", n_var=10, n_obj=3)

    # 3. Создаем оптимизатор
    runner = BotorchRunner(problem, acq_type="qEHVI", device=device)

    # 4. Начальная инициализация
    print("Initializing initial data...")
    runner.initialize_data(n=10)

    # 5. Основной цикл
    n_iterations = 30
    print(f"Starting optimization for {n_iterations} iterations...")

    for i in range(n_iterations):
        # Напоминаю: n_points=1 делает qEHVI эквивалентным классическому EHVI
        candidates, new_y = runner.run_iteration(n_points=1)
        current_best = runner.train_y.max(dim=0).values.tolist()
        formatted_best = [round(val, 4) for val in current_best]
        print(f"Iteration {i + 1}/{n_iterations} completed. Best Y (max): {formatted_best}")

    # 6. Визуализация результатов
    # Инвертируем Y обратно для красивого графика
    res_y = -runner.train_y.detach().cpu().numpy()

    # Истинный фронт из pymoo для сравнения
    pf = problem.pymoo_problem.pareto_front()

    plot = Scatter(title="MOBO Results: DTLZ2 (3 Objectives)", labels=["f1", "f2", "f3"])
    if pf is not None:
        plot.add(pf, plot_type="line", color="black", alpha=0.3, label="True PF")
    plot.add(res_y, color="red", label="Bayesian Opt (qEHVI)")
    plot.show()

if __name__ == "__main__":
    main()