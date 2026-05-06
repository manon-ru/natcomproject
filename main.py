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
from visualization.plot import visualize_maze

# ---------------------------------------------------------------------------
# Maze types
# ---------------------------------------------------------------------------
TIERS = [
    {"name": "U-Trap (Deception)",     "width": 10, "height": 10, "seed": 2026, "type": "U-Trap"},
    {"name": "Sudden Wall (Dynamic)",  "width": 15, "height": 15, "seed": 2026, "type": "Sudden Wall"},
    {"name": "Parallel (Multimodal)", "width": 15, "height": 15, "seed": 2026, "type": "Parallel Paths"},
]

# Number of independent trials per algorithm per tier (proposal: 50)
NUM_TRIALS = 50
LIMIT_MULTIPLIER = 3


def run_trials(AlgorithmClass, maze, num_trials: int, max_iterations: int, **kwargs) -> tuple[list[dict], list[float]]:
    """Run AlgorithmClass for num_trials independent trials on maze."""
    results = []
    entropy_history = []

    for _ in range(num_trials):
        algo = AlgorithmClass(maze, **kwargs)
        result = algo.run(max_iterations=max_iterations)
        results.append(result)
        # Collect entropy history from the last trial for plotting
        if hasattr(algo, "entropy_history"):
            entropy_history = algo.entropy_history

    return results, entropy_history


def print_summary(tier_name: str, algo_name: str, results: list[dict], maze, entropy_history: list[float]) -> None:
    sr = success_rate(results)
    mic = mean_iteration_count(results)
    
    # 1. Calculate path lengths for all successful trials
    # We subtract 1 because len(path) is the number of nodes, and steps = nodes - 1
    successful_steps = [len(r["path"]) - 1 for r in results if r["success"] and r["path"]]
    
    # 2. Find the best (minimum) number of steps found across all trials
    best_found = min(successful_steps) if successful_steps else None
    
    # 3. Calculate mean optimality for successes
    opt_values = [path_optimality(r, maze) for r in results if r["success"] and r["path"]]
    mean_opt = sum(opt_values) / len(opt_values) if opt_values else None
    
    dlr = diversity_loss_rate(entropy_history)
    optimal = shortest_path_length(maze) - 1  # Optimal steps is shortest path length minus 1 (steps = nodes - 1)

    print(f"  [{algo_name}]")
    print(f"    Success rate:       {sr:.0%} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"    Mean iterations:    {mic:.1f}" if mic != float("inf") else "    Mean iterations:    N/A (no successes)")
    
    # Print the Best steps vs Optimal steps
    if best_found is not None:
        color_code = "\033[92m" if best_found == optimal else "" # Green if optimal
        reset_code = "\033[0m"
        print(f"    Best found steps:   {color_code}{best_found}{reset_code} (Optimal: {optimal})")
    else:
        print(f"    Best found steps:   N/A (Optimal: {optimal})")
        
    print(f"    Path optimality:    {mean_opt:.3f}" if mean_opt is not None else "    Path optimality:    N/A")
    print(f"    Diversity loss/10i: {dlr:.4f}" if dlr is not None else "    Diversity loss/10i: N/A")
    print()


def main() -> None:
    for tier in TIERS:
        print(f"=== Testing: {tier['name']} ===")
        # Use the specific maze_type from our TIERS list
        maze = generate_maze(tier["width"], tier["height"], seed=tier["seed"], maze_type=tier["type"])

        optimal_len = shortest_path_length(maze)

        if optimal_len is None:
            print("  WARNING: Maze is unsolvable! Skipping this tier.")
            continue
        
        dynamic_limit = int(optimal_len * LIMIT_MULTIPLIER)
        
        print(f"Optimal path: {optimal_len} steps | Max iterations: {dynamic_limit}")
        print("-" * 50)

        # Baseline tests as currently implemented
        algorithms_to_run = [
            ("Baseline (DFS)", BaselineGA, {"backtrack": True}),
            ("Baseline (Naive)", BaselineGA, {"backtrack": False}),
        ]

        for algo_name, AlgoClass, kwargs in algorithms_to_run:
            results, entropy_history = run_trials(
                AlgoClass, 
                maze, 
                NUM_TRIALS, 
                max_iterations=dynamic_limit, 
                **kwargs
            )
            
            print_summary(tier["name"], algo_name, results, maze, entropy_history)
            
            # Find and visualize best attempt
            best_path = None
            best_dist = float('inf')
            for r in results:
                path = r["path"]
                if not path: continue
                last_pos = path[-1]
                dist = abs(last_pos[0] - maze.goal[0]) + abs(last_pos[1] - maze.goal[1])
                if dist < best_dist or (dist == best_dist and len(path) < len(best_path or [])):
                    best_dist = dist
                    best_path = path

            status = "SUCCESS" if best_dist == 0 else f"FAILED (dist={best_dist})"
            visualize_maze(
                maze, 
                path=best_path, 
                title=f"{algo_name} on {tier['name']} [{status}]", 
                show=True
            )
        print()

if __name__ == "__main__":
    main()