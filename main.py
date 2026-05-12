"""
main.py
Experimental runner for Group 27 Natural Computing project.
"""

import sys
import os
import random
import numpy as np

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import PSO
from algorithms.aco import ACO
from evaluation.metrics import (
    success_rate,
    mean_iteration_count,
    time_to_half_entropy,
    diversity_floor,
    mean_entropy,
    adaptation_time
)
from visualization.plot import visualize_maze

# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

NUM_TRIALS = 10
NUM_SEEDS = 10      
START_SEED = 1
LIMIT_MULTIPLIER = 10 
POPULATION_SIZE = 100  
DISRUPTION_TIME_SUDDEN_WALL = 150
RECOVERY_THRESHOLD = 0.5

WIDTH = 40
HEIGHT = 40

SHOW_ALL_PATHS = True  
PLOT_MAZE = False 

TIERS = [
    # {"name": "U-Trap (Deception)",     "width": WIDTH, "height": HEIGHT, "type": "U-Trap"},
    {"name": "Sudden Wall (Dynamic)",  "width": WIDTH, "height": HEIGHT, "type": "Sudden Wall"},
    # {"name": "Parallel (Multimodal)",  "width": WIDTH, "height": HEIGHT, "type": "Parallel Paths"},
]

ALGORITHMS_TO_RUN = [
    ("Genetic Algorithm", GeneticAlgorithm, {"pop_size": POPULATION_SIZE, "tournament_k": 5}),
    ("Particle Swarm Optimization", PSO, {"num_particles": POPULATION_SIZE, "c1": 0.1, "c2": 0.2}),
    ("Ant Colony Optimization", ACO, {"num_ants": POPULATION_SIZE, "evaporation_rate": 0.1}),
]

def run_trials(AlgorithmClass, maze, num_trials: int, max_iterations: int, disruption_iteration: int = -1, **kwargs) -> tuple[list[dict], list[list[float]]]:
    results = []
    all_entropy_histories = []
    
    # We want to see adaptation even after success
    # If it's a dynamic maze, we ensure we run for at least X steps after disruption
    forced_min_iterations = disruption_iteration + 100 if disruption_iteration > 0 else 0
    
    for _ in range(num_trials):
        if hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
            maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
            
        algo = AlgorithmClass(maze, **kwargs)
        
        # Modify this: The algorithm should not 'break' on success 
        # unless it has reached the forced_min_iterations
        result = algo.run(
            max_iterations=max(max_iterations, forced_min_iterations), 
            disruption_iteration=disruption_iteration
        )
        results.append(result)
        
        if hasattr(algo, "entropy_history"):
            all_entropy_histories.append(algo.entropy_history)
            
    return results, all_entropy_histories

def print_summary(
    tier_name: str, 
    algo_name: str, 
    results: list[dict], 
    entropy_histories: list[list[float]], 
    disruption_iteration: int = -1, 
    recovery_ratio: float = 0.8
) -> None:
    """
    Prints a detailed performance summary including raw path lengths 
    and averaged optimality metrics.
    """
    sr = success_rate(results)
    mic = mean_iteration_count(results)
    
    successful_runs = [r for r in results if r["success"] and r["path"]]
    
    if successful_runs:
        actual_lengths = [len(r["path"]) - 1 for r in successful_runs]
        optimal_lengths = [r["optimal_steps"] for r in successful_runs]
        
        avg_optimal_len = sum(optimal_lengths) / len(optimal_lengths)
        best_found_len = min(actual_lengths)
        mean_found_len = sum(actual_lengths) / len(actual_lengths)
        
        # Calculate optimality per run to handle varying seed difficulties fairly
        opt_ratios = [act / opt for act, opt in zip(actual_lengths, optimal_lengths) if opt > 0]
        mean_opt_ratio = sum(opt_ratios) / len(opt_ratios) if opt_ratios else 1.0
        
        optimal_hits = sum(1 for act, opt in zip(actual_lengths, optimal_lengths) if act == opt)
    else:
        avg_optimal_len = best_found_len = mean_found_len = mean_opt_ratio = None
        optimal_hits = 0
    
    # Entropy metrics (Averaged across all trials)
    half_times = [time_to_half_entropy(h) for h in entropy_histories if time_to_half_entropy(h) is not None]
    floors = [diversity_floor(h) for h in entropy_histories if diversity_floor(h) is not None]
    means = [mean_entropy(h) for h in entropy_histories if mean_entropy(h) is not None]
    
    avg_half_time = sum(half_times) / len(half_times) if half_times else None
    avg_floor = sum(floors) / len(floors) if floors else None
    avg_mean_ent = sum(means) / len(means) if means else None
    
    # Adaptation logic
    avg_adapt_time = None
    if disruption_iteration > 0:
        adapt_times = [
            adaptation_time(h, disruption_iteration, threshold_ratio=recovery_ratio) 
            for h in entropy_histories 
            if adaptation_time(h, disruption_iteration, threshold_ratio=recovery_ratio) is not None
        ]
        avg_adapt_time = sum(adapt_times) / len(adapt_times) if adapt_times else None

    # Iteration string
    if mic == float("inf"):
        it_str = "N/A"
    elif disruption_iteration > 0:
        it_str = f"{mic - disruption_iteration:.1f} (post-disruption)"
    else:
        it_str = f"{mic:.1f}"

    # Output
    print(f"  [{algo_name}]")
    print(f"    Success rate:          {sr:.0%} ({len(successful_runs)}/{len(results)})")
    print(f"    Mean iterations:       {it_str}")
    
    # Path Metrics
    if successful_runs:
        print(f"    Optimal steps (Mean):  {avg_optimal_len:.1f}")
        print(f"    Best found steps:      {best_found_len}")
        print(f"    Mean found steps:      {mean_found_len:.1f}")
        print(f"    Optimal paths found:   {optimal_hits}/{len(successful_runs)} of successful runs")
        print(f"    Path optimality:       {mean_opt_ratio:.3f}")
    else:
        print(f"    Path metrics:          N/A")
        
    print(f"    Time to 50% entropy:   {avg_half_time:.1f} iterations" if avg_half_time is not None else "    Time to 50% entropy:   N/A")
    print(f"    Diversity floor:       {avg_floor:.4f} bits" if avg_floor is not None else "    Diversity floor:       N/A")
    print(f"    Mean entropy:          {avg_mean_ent:.4f} bits" if avg_mean_ent is not None else "    Mean entropy:          N/A")
    
    if disruption_iteration > 0:
        label = f"{recovery_ratio:.0%} Adaptation time"
        print(f"    {label}:   {avg_adapt_time:.1f} iterations" if avg_adapt_time is not None else f"    {label}:   N/A")
    print()
    
def main() -> None:
    for tier in TIERS:
        print(f"=== Testing: {tier['name']} (Seeds {START_SEED} to {START_SEED + NUM_SEEDS - 1}) ===")
        print("-" * 50)
        
        algo_data = {algo_name: {"results": [], "entropy": []} for algo_name, _, _ in ALGORITHMS_TO_RUN}
        current_disruption = DISRUPTION_TIME_SUDDEN_WALL if tier["type"] == "Sudden Wall" else -1

        for seed in range(START_SEED, START_SEED + NUM_SEEDS):
            maze = generate_maze(tier["width"], tier["height"], seed=seed, maze_type=tier["type"])
            
            # Reset seeds for trial stochasticity
            random.seed()
            np.random.seed()
            
            initial_optimal_path = maze.shortest_path()
            if initial_optimal_path is None: continue
            initial_optimal_steps = len(initial_optimal_path) - 1

            # Determine true optimal path (after disruption if it exists)
            true_optimal_path = initial_optimal_path
            if hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
                maze.add_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
                true_optimal_path = maze.shortest_path()
                maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
                true_optimal_steps = len(true_optimal_path) - 1
            else:
                true_optimal_steps = initial_optimal_steps

            dynamic_limit = int(true_optimal_steps * LIMIT_MULTIPLIER)

            for algo_name, AlgoClass, kwargs in ALGORITHMS_TO_RUN:
                results, all_entropy_histories = run_trials(
                    AlgoClass, maze, NUM_TRIALS, 
                    max_iterations=dynamic_limit, 
                    disruption_iteration=current_disruption, 
                    **kwargs
                )
                
                # Tag results with the optimal steps for this specific seed
                for r in results:
                    r["optimal_steps"] = true_optimal_steps
                    
                algo_data[algo_name]["results"].extend(results)
                algo_data[algo_name]["entropy"].extend(all_entropy_histories)
                
                # Visualization logic for the first seed of each tier
                if PLOT_MAZE and seed == START_SEED:
                    all_trial_history = []   
                    all_trial_snapshots = [] 
                    best_disruption_iteration = None

                    # Find the best path actually achieved by the algorithm in this set of trials
                    successful_paths = [r["path"] for r in results if r["success"] and r["path"]]
                    best_run_path = min(successful_paths, key=len) if successful_paths else None

                    for r in results:
                        if SHOW_ALL_PATHS and "history" in r: 
                            all_trial_history.append(r["history"])
                        if r.get("snapshot_history") and SHOW_ALL_PATHS: 
                            all_trial_snapshots.append(r["snapshot_history"])
                        if best_disruption_iteration is None: 
                            best_disruption_iteration = r.get("disruption_iteration")

                    # 1. Visualization for the Sudden Wall disruption point (T=100)
                    if tier["type"] == "Sudden Wall" and hasattr(maze, 'dynamic_wall') and maze.dynamic_wall:
                        maze.remove_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])
                        visualize_maze(
                            maze, 
                            optimal_path=initial_optimal_path, 
                            all_paths=all_trial_snapshots if all_trial_snapshots else all_trial_history,
                            title=f"{algo_name} at T={best_disruption_iteration or '?'}", 
                            show=True
                        )
                        maze.add_wall(maze.dynamic_wall[0], maze.dynamic_wall[1])

                    # 2. Final Result Visualization (Red = Optimal, Green = Best Found)
                    visualize_maze(
                        maze, 
                        optimal_path=true_optimal_path, 
                        best_found_path=best_run_path,
                        all_paths=all_trial_history,
                        title=f"{algo_name} Final (Red=Opt, Green=Best Found) [Seed {seed}]", 
                        show=True
                    )

        # Print aggregate summary across all seeds/trials for the current Tier
        for algo_name, _, _ in ALGORITHMS_TO_RUN:
            print_summary(
                tier["name"], 
                algo_name, 
                algo_data[algo_name]["results"], 
                algo_data[algo_name]["entropy"], 
                current_disruption, 
                recovery_ratio=RECOVERY_THRESHOLD
            )
                    
if __name__ == "__main__":
    main()
