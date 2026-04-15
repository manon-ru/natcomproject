from collections import deque

from maze.environment import MazeEnvironment


def shortest_path_length(maze: MazeEnvironment) -> int | None:
    """
    BFS to find the length of the shortest path from start to goal.
    Returns the number of cells in the path (including start and goal),
    or None if the goal is unreachable.
    """
    queue: deque[tuple[tuple, int]] = deque([(maze.start, 1)])
    visited = {maze.start}

    while queue:
        (x, y), length = queue.popleft()
        if (x, y) == maze.goal:
            return length
        for nx, ny in maze.neighbors(x, y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), length + 1))

    return None  # Goal unreachable


def success_rate(results: list[dict]) -> float:
    """
    Fraction of runs where the algorithm found the goal.

    Args:
        results: List of result dicts from algorithm.run(), each with key "success".

    Returns:
        Float in [0, 1].
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)


def mean_iteration_count(results: list[dict]) -> float:
    """
    Mean number of iterations across successful runs.

    Args:
        results: List of result dicts with keys "success" and "iterations".

    Returns:
        Mean iteration count, or float('inf') if no successful runs.
    """
    successful = [r["iterations"] for r in results if r["success"]]
    if not successful:
        return float("inf")
    return sum(successful) / len(successful)


def path_optimality(result: dict, maze: MazeEnvironment) -> float | None:
    """
    Ratio of the found path length to the optimal (BFS) path length.
    A value of 1.0 is optimal; higher values indicate suboptimal paths.

    Args:
        result: A single result dict with keys "success" and "path".
        maze:   The MazeEnvironment the algorithm ran on.

    Returns:
        Ratio (found / optimal), or None if the run failed or BFS finds no path.
    """
    if not result["success"] or result["path"] is None:
        return None
    optimal = shortest_path_length(maze)
    if optimal is None:
        return None
    return len(result["path"]) / optimal


def diversity_loss_rate(entropy_history: list[float]) -> float | None:
    """
    Linear slope of entropy over time (bits per sampling interval).
    A negative slope indicates diversity is being lost as the algorithm runs.

    Args:
        entropy_history: Entropy values sampled at regular intervals.

    Returns:
        Slope of a linear fit, or None if fewer than 2 data points.
    """
    if len(entropy_history) < 2:
        return None

    n = len(entropy_history)
    x_mean = (n - 1) / 2
    y_mean = sum(entropy_history) / n

    numerator = sum((i - x_mean) * (h - y_mean) for i, h in enumerate(entropy_history))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return 0.0
    return numerator / denominator
