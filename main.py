"""
main.py — Experimental driver for Group 27 NatComp project.

Runs the full 3 × 3 × 3 factorial (algorithm × maze type × population size)
with 10 instances × 10 trials per cell = 2,700 total runs.

Usage:
    uv run python main.py            # Full 2,700-run experiment → results/runs.csv
    uv run python main.py --quick    # Smoke test: 9 runs → results/runs_quick.csv
    uv run python main.py --pilot    # 27-cell pilot: 27 runs → results/runs_pilot.csv
"""
import sys
import os
import time
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import (
    POPULATION_SIZES,
    MAZE_TYPES,
    INSTANCE_SEEDS,
    NUM_TRIALS,
    RESULTS_CSV_PATH,
)
from runner import run_experiment
from results_writer import ResultsWriter

ALGORITHMS = ["GA", "PSO", "ACO"]


def build_tasks(quick: bool = False, pilot: bool = False) -> list:
    """Build the list of (algo, maze_type, pop_size, instance_seed, trial_seed) tasks."""
    tasks = []
    if quick:
        # 3 algos × 3 mazes × 1 pop × 1 instance × 1 trial = 9 tasks
        for algo in ALGORITHMS:
            for maze in MAZE_TYPES:
                tasks.append((algo, maze, 20, 1, 42))
    elif pilot:
        # 3 algos × 3 mazes × 3 pops × 1 instance × 1 trial = 27 tasks
        for algo in ALGORITHMS:
            for maze in MAZE_TYPES:
                for pop in POPULATION_SIZES:
                    tasks.append((algo, maze, pop, 1, 42))
    else:
        # Full factorial: 3 × 3 × 3 × 10 × 10 = 2,700 tasks
        for algo in ALGORITHMS:
            for maze in MAZE_TYPES:
                for pop in POPULATION_SIZES:
                    for instance in INSTANCE_SEEDS:
                        for trial in range(NUM_TRIALS):
                            trial_seed = instance * 1000 + trial
                            tasks.append((algo, maze, pop, instance, trial_seed))
    return tasks


def main() -> None:
    quick = "--quick" in sys.argv
    pilot = "--pilot" in sys.argv

    if quick:
        csv_path = "results/runs_quick.csv"
    elif pilot:
        csv_path = "results/runs_pilot.csv"
    else:
        csv_path = RESULTS_CSV_PATH

    tasks = build_tasks(quick=quick, pilot=pilot)
    total = len(tasks)
    num_workers = min(os.cpu_count() or 4, 8)

    print(f"Mode: {'quick' if quick else 'pilot' if pilot else 'full'}")
    print(f"Total tasks: {total}")
    print(f"Workers: {num_workers}")
    print(f"Output: {csv_path}")

    multiprocessing.set_start_method("spawn", force=True)

    start = time.time()
    done = 0

    with ResultsWriter(csv_path) as writer:
        def on_done(result):
            nonlocal done
            writer.write(result)
            done += 1
            if done % 50 == 0 or done == total:
                dt = time.time() - start
                eta = dt / done * (total - done) if done < total else 0.0
                print(f"[{done}/{total}] elapsed={dt:.1f}s eta={eta:.1f}s")

        run_experiment(tasks, num_workers=num_workers, on_complete=on_done)

    print(f"Done. {done}/{total} results written to {csv_path}")


if __name__ == "__main__":
    main()
