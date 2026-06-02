"""
Experimental driver for the project.

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
from tqdm import tqdm

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
    requested = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--workers=")), None)
    num_workers = requested if requested else (os.cpu_count() or 4)

    print(f"Mode: {'quick' if quick else 'pilot' if pilot else 'full'}")
    print(f"Total tasks: {total}")
    print(f"Workers: {num_workers}")
    print(f"Output: {csv_path}")

    multiprocessing.set_start_method("spawn", force=True)

    start = time.time()

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    interrupted = False
    with ResultsWriter(csv_path) as writer:
        with tqdm(total=total, unit="run", bar_format=bar_fmt, dynamic_ncols=True) as bar:
            def on_done(result):
                writer.write(result)
                algo = result.get("algo", "?")
                maze = result.get("maze_type", "?")
                bar.set_postfix(algo=algo, maze=maze, refresh=False)
                bar.update(1)

            try:
                run_experiment(tasks, num_workers=num_workers, on_complete=on_done)
            except KeyboardInterrupt:
                interrupted = True
                bar.close()

    dt = time.time() - start
    if interrupted:
        print(f"Interrupted. Partial results written to {csv_path} after {dt:.1f}s")
        sys.exit(130)
    print(f"Done. {total} results written to {csv_path} in {dt:.1f}s")


if __name__ == "__main__":
    main()
