"""
main.py
Experimental runner for Group 27 Natural Computing project.
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
    diversity_loss_rate,
)
from visualization.plot import visualize_maze

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

NUM_TRIALS = 10
LIMIT_MULTIPLIER = 10 
SHOW_ALL_PATHS = True  
ENABLE_PLOTTING = True 
POPULATION_SIZE = 100  
DISRUPTION_TIME = 100

TIERS = [
    # {"name": "U-Trap (Deception)",     "width": 10, "height": 10, "seed": 1, "type": "U-Trap"},
    {"name": "Sudden Wall (Dynamic)",  "width": 30, "height": 30, "seed": 2026, "type": "Sudden Wall"},
    # {"name": "Parallel (Multimodal)",  "width": 15, "height": 15, "seed": 2026, "type": "Parallel Paths"},
]

ALGORITHMS_TO_RUN = [
    ("Baseline (Backtrack)", BaselineGA, {"backtrack": True}),
    ("Genetic Algorithm", GeneticAlgorithm, {"pop_size": POPULATION_SIZE, "tournament_k": 5}),
    ("Particle Swarm Optimization", PSO, {"num_particles": POPULATION_SIZE, "c1": 0.1, "c2": 0.2}),
    ("Ant Colony Optimization", ACO, {"num_ants": POPULATION_SIZE, "evaporation_rate": 0.1}),
]

def run_trials(AlgorithmClass, maze, num_trials: int, max_iterations: int, disruption_iteration: int = -1, **kwargs) -> tuple[list[dict], list[float]]:
    results = []
    entropy_history = []
    for _ in range(num_trials):
        if hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
            maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
        algo = AlgorithmClass(maze, **kwargs)
        result = algo.run(max_iterations=max_iterations, disruption_iteration=disruption_iteration)
        results.append(result)
        if hasattr(algo, "entropy_history"):
            entropy_history = algo.entropy_history
    return results, entropy_history

def print_summary(tier_name: str, algo_name: str, results: list[dict], entropy_history: list[float], optimal_steps: int, disruption_iteration: int = -1) -> None:
    sr = success_rate(results)
    mic = mean_iteration_count(results)
    
    successful_steps = [len(r["path"]) - 1 for r in results if r["success"] and r["path"]]
    best_found = min(successful_steps) if successful_steps else None
    
    if successful_steps:
        opt_values = [steps / optimal_steps for steps in successful_steps]
        mean_opt = sum(opt_values) / len(opt_values)
    else:
        mean_opt = None
    
    dlr = diversity_loss_rate(entropy_history)

    # Calculate the iteration string
    if mic == float("inf"):
        it_str = "N/A"
    elif disruption_iteration > 0:
        # Show iterations spent strictly AFTER the settlement phase
        # Recovery Time = $Total\ Iterations - Disruption\ Threshold$
        recovery_time = mic - disruption_iteration
        it_str = f"{recovery_time:.1f} (after wall drop)"
    else:
        # Standard display for static mazes
        it_str = f"{mic:.1f}"

    print(f"  [{algo_name}]")
    print(f"    Success rate:       {sr:.0%} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"    Mean iterations:    {it_str}")
    
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
        initial_optimal_path = maze.shortest_path()
        if initial_optimal_path is None: continue
        initial_optimal_steps = len(initial_optimal_path) - 1

        true_optimal_path = initial_optimal_path
        if hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
            maze.add_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
            true_optimal_path = maze.shortest_path()
            maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
            true_optimal_steps = len(true_optimal_path) - 1
        else:
            true_optimal_steps = initial_optimal_steps

        dynamic_limit = int(true_optimal_steps * LIMIT_MULTIPLIER)
        
        # GATE: Only use the disruption threshold for Sudden Wall mazes
        current_disruption = DISRUPTION_TIME if tier["type"] == "Sudden Wall" else -1
        
        print(f"Initial path: {initial_optimal_steps} steps | True Optimal (with wall): {true_optimal_steps} steps")
        if current_disruption > 0:
            print(f"Trigger drop at iteration: {current_disruption}")
        print(f"Max iterations: {dynamic_limit}")
        print("-" * 50)

        for algo_name, AlgoClass, kwargs in ALGORITHMS_TO_RUN:
            results, entropy_history = run_trials(
                AlgoClass, maze, NUM_TRIALS, 
                max_iterations=dynamic_limit, 
                disruption_iteration=current_disruption, 
                **kwargs
            )
            print_summary(tier["name"], algo_name, results, entropy_history, true_optimal_steps, current_disruption)
            
            if ENABLE_PLOTTING:
                all_trial_snapshots = [] 
                all_trial_history = []   
                best_disruption_iteration = None
                for r in results:
                    if SHOW_ALL_PATHS and "history" in r: all_trial_history.append(r["history"])
                    if r.get("snapshot_history") and SHOW_ALL_PATHS: all_trial_snapshots.append(r["snapshot_history"])
                    if best_disruption_iteration is None: best_disruption_iteration = r.get("disruption_iteration")

                if tier["type"] == "Sudden Wall" and hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
                    maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
                    visualize_maze(maze, path=initial_optimal_path, all_paths=all_trial_snapshots,
                                   title=f"{algo_name} at T={best_disruption_iteration or '?'}", show=True)
                    maze.add_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])

                visualize_maze(maze, path=true_optimal_path, all_paths=all_trial_history,
                               title=f"{algo_name} Final Result (Red=Optimal)", show=True)
        print()

if __name__ == "__main__":
    main()