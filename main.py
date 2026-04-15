"""
main.py — Experimental runner for Group 27 Natural Computing project.

Runs all four algorithms (BaselineGA, GA, PSO, ACO) across three maze tiers
and prints a summary of results. Visualizes the maze for each tier.

Usage:
    uv run main.py
"""

import sys
import os

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maze.generator import generate_maze
from algorithms.baseline_ga import BaselineGA
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import PSO
from algorithms.aco import ACO
from evaluation.metrics import (
    success_rate,
    mean_iteration_count,
    path_optimality,
    diversity_loss_rate,
    shortest_path_length,
)
from visualization.plot import visualize_maze, visualize_entropy

# ---------------------------------------------------------------------------
# Maze tiers (from proposal Section 3.4)
# ---------------------------------------------------------------------------
TIERS = [
    {"name": "Simple",  "width": 5,  "height": 5,  "seed": 2026},
    {"name": "Medium",  "width": 10, "height": 10, "seed": 2026},
    {"name": "Complex", "width": 20, "height": 20, "seed": 2026},
]

# Number of independent trials per algorithm per tier (proposal: 50)
NUM_TRIALS = 50
MAX_ITERATIONS = 1000


def run_trials(AlgorithmClass, maze, num_trials: int, **kwargs) -> tuple[list[dict], list[float]]:
    """Run AlgorithmClass for num_trials independent trials on maze."""
    results = []
    entropy_history = []

    for _ in range(num_trials):
        algo = AlgorithmClass(maze, **kwargs)
        result = algo.run(max_iterations=MAX_ITERATIONS)
        results.append(result)
        # Collect entropy history from the last trial for plotting
        if hasattr(algo, "entropy_history"):
            entropy_history = algo.entropy_history

    return results, entropy_history


def print_summary(tier_name: str, algo_name: str, results: list[dict], maze, entropy_history: list[float]) -> None:
    sr = success_rate(results)
    mic = mean_iteration_count(results)
    opt_values = [path_optimality(r, maze) for r in results if r["success"] and r["path"]]
    mean_opt = sum(opt_values) / len(opt_values) if opt_values else None
    dlr = diversity_loss_rate(entropy_history)
    optimal = shortest_path_length(maze)

    print(f"  [{algo_name}]")
    print(f"    Success rate:       {sr:.0%} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"    Mean iterations:    {mic:.1f}" if mic != float("inf") else "    Mean iterations:    N/A (no successes)")
    print(f"    Path optimality:    {mean_opt:.3f}" if mean_opt is not None else "    Path optimality:    N/A")
    print(f"    Optimal path len:   {optimal}")
    print(f"    Diversity loss/10i: {dlr:.4f}" if dlr is not None else "    Diversity loss/10i: N/A")
    print()


def main() -> None:
    for tier in TIERS:
        print(f"=== Tier: {tier['name']} ({tier['width']}x{tier['height']}, seed={tier['seed']}) ===")
        maze = generate_maze(tier["width"], tier["height"], seed=tier["seed"])

        # Visualize the maze (without blocking — set show=False to skip display)
        visualize_maze(maze, title=f"{tier['name']} Maze ({tier['width']}x{tier['height']})", show=False)

        entropy_histories: dict[str, list[float]] = {}

        algorithms = [
            ("Baseline GA", BaselineGA, {}),
            ("GA",          GeneticAlgorithm, {"pop_size": 50, "tournament_k": 5}),
            ("PSO",         PSO,              {"num_particles": 30}),
            ("ACO",         ACO,              {}),
        ]

        for algo_name, AlgoClass, kwargs in algorithms:
            results, entropy_history = run_trials(AlgoClass, maze, NUM_TRIALS, **kwargs)
            entropy_histories[algo_name] = entropy_history
            print_summary(tier["name"], algo_name, results, maze, entropy_history)

        # Plot entropy comparison for this tier
        visualize_entropy(
            entropy_histories,
            title=f"Population Diversity — {tier['name']} Maze",
            show=False,
        )

        print()


if __name__ == "__main__":
    main()
