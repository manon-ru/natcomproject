from collections import deque
from math import log2
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

    return None


def success_rate(results: list[dict]) -> float:
    """Fraction of runs where the algorithm found the goal."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["success"]) / len(results)


def mean_iteration_count(results: list[dict]) -> float:
    """Mean number of iterations across successful runs."""
    successful = [r["iterations"] for r in results if r["success"]]
    if not successful:
        return float("inf")
    return sum(successful) / len(successful)


def path_optimality(result: dict, maze: MazeEnvironment) -> float | None:
    """Ratio of the found path length to the optimal (BFS) path length."""
    if not result["success"] or result["path"] is None:
        return None
    optimal = shortest_path_length(maze)
    if optimal is None:
        return None
    return len(result["path"]) / optimal


def calculate_shannon_entropy(positions: list[tuple]) -> float:
    """Compute the Shannon entropy of a population based on agent positions."""
    if not positions:
        return 0.0

    counts: dict[tuple, int] = {}
    for pos in positions:
        counts[pos] = counts.get(pos, 0) + 1

    n = len(positions)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * log2(p)

    return entropy


def time_to_half_entropy(entropy_history: list[float]) -> int | None:
    """
    Iterations until entropy falls to 50% of its peak value.
    Measures the speed of convergence after the initial exploration phase.
    """
    if not entropy_history:
        return None
    
    peak_entropy = max(entropy_history)
    if peak_entropy == 0.0:
        return None
        
    peak_index = entropy_history.index(peak_entropy)
    threshold = peak_entropy * 0.5
    
    for i in range(peak_index, len(entropy_history)):
        if entropy_history[i] <= threshold:
            return i
            
    return None


def diversity_floor(entropy_history: list[float]) -> float | None:
    """
    The minimum entropy reached after the peak exploration phase.
    Measures the severity of convergence.
    """
    if not entropy_history:
        return None
        
    peak_entropy = max(entropy_history)
    if peak_entropy == 0.0:
        return 0.0
        
    peak_index = entropy_history.index(peak_entropy)
    return min(entropy_history[peak_index:])


def mean_entropy(entropy_history: list[float]) -> float | None:
    """The average entropy sustained across the entire run."""
    if not entropy_history:
        return None
    return sum(entropy_history) / len(entropy_history)


def adaptation_time(
    entropy_history: list[float], 
    disruption_iteration: int, 
    threshold_ratio: float = 0.8,
    sample_interval: int = 10
) -> int | None:
    """
    Iterations from disruption until entropy returns to a percentage (threshold_ratio)
    of its pre-disruption value, accounting for the sampling interval.
    """
    # Convert raw iteration to the correct index in the sampled array
    disruption_idx = disruption_iteration // sample_interval
    
    if disruption_idx < 1 or disruption_idx >= len(entropy_history):
        return None
        
    pre_disruption_entropy = entropy_history[disruption_idx - 1]
    
    # If entropy was already 0, any recovery is 100%
    if pre_disruption_entropy == 0:
        threshold = 0.0001 
    else:
        threshold = pre_disruption_entropy * threshold_ratio
    
    # Search the array starting from the disruption point
    for i in range(disruption_idx, len(entropy_history)):
        if entropy_history[i] >= threshold:
            # Convert the array index difference back into actual iterations
            return (i - disruption_idx) * sample_interval
            
    return None