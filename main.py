"""
main.py
Experimental runner for Group 27 Natural Computing project.

Runs algorithms across three maze tiers, handles dynamic disruptions,
prints a summary of results, and visualizes the paths.

Usage:
    uv run main.py
"""

import sys
import os

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maze.generator import generate_maze
from algorithms.baseline_ga import BaselineGA
# from algorithms.ga import GeneticAlgorithm
# from algorithms.pso import PSO
# from algorithms.aco import ACO
from evaluation.metrics import (
    success_rate,
    mean_iteration_count,
    diversity_loss_rate,
)
from visualization.plot import visualize_maze

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------
TIERS = [
    # {"name": "U-Trap (Deception)",     "width": 10, "height": 10, "seed": 2026, "type": "U-Trap"},
    # {"name": "Sudden Wall (Dynamic)",  "width": 15, "height": 15, "seed": 2026, "type": "Sudden Wall"},
    {"name": "Parallel (Multimodal)",  "width": 15, "height": 15, "seed": 2026, "type": "Parallel Paths"},
]

ALGORITHMS_TO_RUN = [
    ("Baseline (DFS)", BaselineGA, {"backtrack": True}),
    ("Baseline (Naive)", BaselineGA, {"backtrack": False}),
]

NUM_TRIALS = 50
LIMIT_MULTIPLIER = 5
SHOW_ALL_PATHS = True  # Toggle this to overlay all trials as a heatmap


def run_trials(AlgorithmClass, maze, num_trials: int, max_iterations: int, disruption_length: int = -1, **kwargs) -> tuple[list[dict], list[float]]:
    """Run AlgorithmClass for num_trials independent trials on maze."""
    results = []
    entropy_history = []

    for _ in range(num_trials):
        # RESET THE DYNAMIC WALL BEFORE EACH TRIAL
        if hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
            maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])

        algo = AlgorithmClass(maze, **kwargs)
        result = algo.run(max_iterations=max_iterations, disruption_length=disruption_length)
        results.append(result)
        
        # Collect entropy history from the last trial for plotting
        if hasattr(algo, "entropy_history"):
            entropy_history = algo.entropy_history

    return results, entropy_history


def print_summary(tier_name: str, algo_name: str, results: list[dict], entropy_history: list[float], optimal_steps: int) -> None:
    sr = success_rate(results)
    mic = mean_iteration_count(results)
    
    # Calculate path lengths for all successful trials
    successful_steps = [len(r["path"]) - 1 for r in results if r["success"] and r["path"]]
    
    # Find the best (minimum) number of steps found across all trials
    best_found = min(successful_steps) if successful_steps else None
    
    # Calculate mean optimality for successes
    if successful_steps:
        opt_values = [steps / optimal_steps for steps in successful_steps]
        mean_opt = sum(opt_values) / len(opt_values)
    else:
        mean_opt = None
    
    dlr = diversity_loss_rate(entropy_history)

    print(f"  [{algo_name}]")
    print(f"    Success rate:       {sr:.0%} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"    Mean iterations:    {mic:.1f}" if mic != float("inf") else "    Mean iterations:    N/A (no successes)")
    
    # Print the Best steps vs Optimal steps
    if best_found is not None:
        color_code = "\033[92m" if best_found == optimal_steps else ""
        reset_code = "\033[0m"
        print(f"    Best found steps:   {color_code}{best_found}{reset_code} (Optimal: {optimal_steps})")
    else:
        print(f"    Best found steps:   N/A (Optimal: {optimal_steps})")
        
    print(f"    Path optimality:    {mean_opt:.3f}" if mean_opt is not None else "    Path optimality:    N/A")
    print(f"    Diversity loss/10i: {dlr:.4f}" if dlr is not None else "    Diversity loss/10i: N/A")
    print()


def main() -> None:
    for tier in TIERS:
        print(f"=== Testing: {tier['name']} ===")
        maze = generate_maze(tier["width"], tier["height"], seed=tier["seed"], maze_type=tier["type"])

        # 1. Calculate Initial Optimal (for the spatial trigger)
        initial_path = maze.shortest_path()
        if initial_path is None:
            print("  WARNING: Maze is unsolvable. Skipping this tier.")
            continue
            
        initial_optimal_steps = len(initial_path) - 1
        disruption_length = int(initial_optimal_steps * 0.8)

        # 2. Calculate True Optimal (for limits and metrics)
        if hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
            maze.add_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
            true_path = maze.shortest_path()
            maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
            if true_path is None:
                print("  WARNING: Maze is unsolvable after wall drops. Skipping this tier.")
                continue
            true_optimal_steps = len(true_path) - 1
        else:
            true_optimal_steps = initial_optimal_steps

        dynamic_limit = int(true_optimal_steps * LIMIT_MULTIPLIER)
        
        print(f"Initial path: {initial_optimal_steps} steps | True Optimal (with wall): {true_optimal_steps} steps")
        print(f"Trigger drop at: {disruption_length} steps | Max iterations: {dynamic_limit}")
        print("-" * 50)

        for algo_name, AlgoClass, kwargs in ALGORITHMS_TO_RUN:
            results, entropy_history = run_trials(
                AlgoClass, maze, NUM_TRIALS, 
                max_iterations=dynamic_limit, 
                disruption_length=disruption_length, 
                **kwargs
            )
            
            print_summary(tier["name"], algo_name, results, entropy_history, true_optimal_steps)
            
            # --- Extract Data for Visualization ---
            best_final_path = None
            best_final_dist = float('inf')
            
            best_snapshot = None
            best_snapshot_dist = float('inf')
            best_disruption_iteration = None
            
            all_trial_paths = []
            all_trial_snapshots = []
            
            for r in results:
                # 1. Process the Final Path
                path = r["path"]
                if path:
                    if SHOW_ALL_PATHS:
                        all_trial_paths.append(path)
                        
                    dist = abs(path[-1][0] - maze.goal[0]) + abs(path[-1][1] - maze.goal[1])
                    if dist < best_final_dist or (dist == best_final_dist and len(path) < len(best_final_path or [])):
                        best_final_dist = dist
                        best_final_path = path

                # 2. Process the Snapshot (Before wall drops)
                snap = r.get("snapshot")
                if snap:
                    if SHOW_ALL_PATHS:
                        all_trial_snapshots.append(snap)
                        
                    snap_dist = abs(snap[-1][0] - maze.goal[0]) + abs(snap[-1][1] - maze.goal[1])
                    if snap_dist < best_snapshot_dist or (snap_dist == best_snapshot_dist and len(snap) < len(best_snapshot or [])):
                        best_snapshot_dist = snap_dist
                        best_snapshot = snap
                        best_disruption_iteration = r.get("disruption_iteration", None)

            # --- DUAL VISUALIZATION ---
            # 1. Plot the "Before" state (if a snapshot exists)
            if best_snapshot and tier["type"] == "Sudden Wall" and hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
                maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
                
                t_val = best_disruption_iteration if best_disruption_iteration is not None else "?"
                
                visualize_maze(
                    maze, 
                    path=best_snapshot, 
                    all_paths=all_trial_snapshots if SHOW_ALL_PATHS else None,
                    title=f"{algo_name} at T={t_val} (Just before wall drops)", 
                    show=True
                )
                maze.add_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])

            # 2. Plot the "After" (Final) state
            # Make sure we use best_final_dist here!
            status = "SUCCESS" if best_final_dist == 0 else f"FAILED (dist={best_final_dist})"
            visualize_maze(
                maze, 
                path=best_final_path, 
                all_paths=all_trial_paths if SHOW_ALL_PATHS else None,
                title=f"{algo_name} Final Result [{status}]", 
                show=True
            )
        print()

if __name__ == "__main__":
    main()