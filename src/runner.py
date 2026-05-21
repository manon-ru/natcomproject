"""
Parallel experiment runner.
Uses ProcessPoolExecutor with spawn start method (caller's responsibility).
Each task is a (algo_name, maze_type, pop_size, instance_seed, trial_seed) tuple.
"""
import sys
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np

# Ensure src/ siblings are importable from worker processes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GA_PARAMS,
    GA_CHROMOSOME_LENGTH_FN,
    PSO_PARAMS,
    ACO_PARAMS,
    MAZE_WIDTH,
    MAZE_HEIGHT,
    DISRUPTION_TIME,
    RECOVERY_THRESHOLD,
    ENTROPY_SAMPLE_INTERVAL,
    ENTROPY_FLOOR_FOR_ADAPTATION,
    ITERATION_LIMIT_MULTIPLIER,
    FORCED_MIN_ITERATIONS_AFTER_DISRUPTION,
)
from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import PSO
from algorithms.aco import ACO
from evaluation.metrics import (
    time_to_half_entropy,
    diversity_floor,
    mean_entropy,
    adaptation_time,
)


def _seed_worker(trial_seed: int) -> None:
    """Seed both random modules after maze generation."""
    random.seed(trial_seed)
    np.random.seed(trial_seed)


def _build_algorithm(algo_name: str, maze, pop_size: int):
    if algo_name == "GA":
        return GeneticAlgorithm(
            maze,
            pop_size=pop_size,
            chromosome_length=GA_CHROMOSOME_LENGTH_FN(maze.width, maze.height),
            crossover_rate=GA_PARAMS["crossover_rate"],
            mutation_rate=GA_PARAMS["mutation_rate"],
        )
    elif algo_name == "PSO":
        return PSO(
            maze,
            num_particles=pop_size,
            omega=PSO_PARAMS["omega"],
            c1=PSO_PARAMS["c1"],
            c2=PSO_PARAMS["c2"],
        )
    elif algo_name == "ACO":
        return ACO(
            maze,
            num_ants=pop_size,
            alpha=ACO_PARAMS["alpha"],
            beta=ACO_PARAMS["beta"],
            pheromone_deposit=ACO_PARAMS["Q"],
            evaporation_rate=ACO_PARAMS["rho"],
            initial_pheromone=ACO_PARAMS["tau0"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name!r}")


def run_single_trial(task: tuple) -> dict:
    """Worker function. Each task is (algo_name, maze_type, pop_size, instance_seed, trial_seed)."""
    algo_name, maze_type, pop_size, instance_seed, trial_seed = task

    # Disruption / forced-min setup
    disruption_iteration = DISRUPTION_TIME if maze_type == "Sudden Wall" else -1
    forced_min = (
        DISRUPTION_TIME + FORCED_MIN_ITERATIONS_AFTER_DISRUPTION
        if maze_type == "Sudden Wall"
        else 0
    )

    # Generate maze (this internally seeds random + numpy with instance_seed)
    maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT, seed=instance_seed, maze_type=maze_type)

    # Compute optimal path lengths
    initial_optimal_path = maze.shortest_path()
    if initial_optimal_path is None:
        return {
            "algo": algo_name, "maze_type": maze_type, "pop_size": pop_size,
            "instance_seed": instance_seed, "trial_seed": trial_seed,
            "success_overall": False, "success_postdisruption": False,
            "iterations": 0, "path_length": None, "optimal_length": None,
            "path_optimality": None, "time_to_half_entropy": None,
            "diversity_floor": None, "mean_entropy": None,
            "adaptation_time": None, "entropy_history": [],
        }

    initial_optimal_steps = len(initial_optimal_path) - 1
    true_optimal_steps = initial_optimal_steps

    if maze_type == "Sudden Wall" and getattr(maze, "dynamic_wall", None):
        w1, w2 = maze.dynamic_wall
        maze.add_wall(w1, w2)
        post_path = maze.shortest_path()
        maze.remove_wall(w1, w2)
        if post_path is not None:
            true_optimal_steps = len(post_path) - 1

    max_iterations = max(
        int(true_optimal_steps * ITERATION_LIMIT_MULTIPLIER),
        forced_min + 50,
    )

    # CRITICAL: Re-seed AFTER generate_maze so algorithm randomness is controlled by trial_seed
    _seed_worker(trial_seed)

    # Build and run algorithm
    algo = _build_algorithm(algo_name, maze, pop_size)
    result = algo.run(
        max_iterations=max_iterations,
        disruption_iteration=disruption_iteration,
        forced_min_iterations=forced_min,
    )

    # Compute metrics
    success_overall = bool(result["success"])
    success_postdisruption = bool(
        result["success"]
        and (disruption_iteration < 0 or result["iterations"] > disruption_iteration)
    )
    path_length = len(result["path"]) - 1 if result.get("path") else None
    path_optimality = (
        path_length / true_optimal_steps
        if (path_length is not None and true_optimal_steps > 0)
        else None
    )

    eh = list(algo.entropy_history) if hasattr(algo, "entropy_history") else []

    tt_half = time_to_half_entropy(eh, sample_interval=ENTROPY_SAMPLE_INTERVAL)
    d_floor = diversity_floor(eh)
    m_ent = mean_entropy(eh)
    adapt = (
        adaptation_time(
            eh,
            disruption_iteration,
            threshold_ratio=RECOVERY_THRESHOLD,
            sample_interval=ENTROPY_SAMPLE_INTERVAL,
            entropy_floor=ENTROPY_FLOOR_FOR_ADAPTATION,
        )
        if disruption_iteration > 0
        else None
    )

    return {
        "algo": algo_name,
        "maze_type": maze_type,
        "pop_size": pop_size,
        "instance_seed": instance_seed,
        "trial_seed": trial_seed,
        "success_overall": success_overall,
        "success_postdisruption": success_postdisruption,
        "iterations": result["iterations"],
        "path_length": path_length,
        "optimal_length": true_optimal_steps,
        "path_optimality": path_optimality,
        "time_to_half_entropy": tt_half,
        "diversity_floor": d_floor,
        "mean_entropy": m_ent,
        "adaptation_time": adapt,
        "entropy_history": eh,
    }


def _kill_pool(executor: ProcessPoolExecutor) -> None:
    """Forcefully terminate all worker processes in a ProcessPoolExecutor."""
    procs = list(getattr(executor, "_processes", {}).values())
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=2)
        except Exception:
            pass
    for p in procs:
        if p.is_alive():
            try:
                p.kill()
            except Exception:
                pass


def run_experiment(
    tasks: list,
    num_workers: int,
    on_complete: Callable[[dict], None],
) -> None:
    """Run tasks in parallel. Calls on_complete(result_dict) per finished task.

    On KeyboardInterrupt, queued tasks are cancelled and running workers are
    terminated immediately instead of being drained.
    """
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = {executor.submit(run_single_trial, task): task for task in tasks}

    try:
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                algo, maze_type, pop_size, inst, trial = task
                print(
                    f"[ERROR] {algo} {maze_type} pop={pop_size} inst={inst} trial={trial}: {exc}",
                    file=sys.stderr,
                )
                continue
            on_complete(result)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Cancelling queued tasks and terminating workers...", file=sys.stderr)
        for f in futures:
            f.cancel()
        _kill_pool(executor)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
