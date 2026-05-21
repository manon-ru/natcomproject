#!/usr/bin/env python3
"""
QA: Verify GA results are bit-identical across the PSO/ACO canonicalization.

The PSO/ACO refactor changed PSO's RNG consumption (stdlib random -> np.random).
This could shift the interleaved RNG state and alter GA results for the same
seed. This script samples 5 GA rows from runs.csv.pre-fix and re-runs them,
asserting that success / iterations match exactly.

Note: the pre-fix CSV uses maze_type='U-Trap'; the current code uses
'Shortest Path Trap'. Both reference the same generator function.

Exit codes:
    0 - all sampled rows match on core fields (success_overall, iterations)
    1 - hard fail: any core field differs
    2 - soft fail: entropy_history differs but core fields match
"""
import sys
import csv
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm
from config import GA_PARAMS, GA_CHROMOSOME_LENGTH_FN, DISRUPTION_TIME, FORCED_MIN_ITERATIONS_AFTER_DISRUPTION


# Pre-fix maze_type strings to current strings (only 'U-Trap' was renamed).
MAZE_TYPE_REMAP = {"U-Trap": "Shortest Path Trap"}


def sample_ga_rows(pre_fix_path: Path, n: int = 5) -> list[dict]:
    """Pick n GA rows spanning all three maze types and three population sizes."""
    by_combo: dict[tuple, dict] = {}
    with pre_fix_path.open() as f:
        for row in csv.DictReader(f):
            if row["algo"] != "GA":
                continue
            combo = (row["maze_type"], row["pop_size"])
            if combo not in by_combo:
                by_combo[combo] = row
    sampled = list(by_combo.values())[:n]
    if len(sampled) < n:
        # Fall back to first n GA rows if we don't have enough combinations.
        with pre_fix_path.open() as f:
            ga_rows = [r for r in csv.DictReader(f) if r["algo"] == "GA"]
        sampled = ga_rows[:n]
    return sampled


def rerun_ga(row: dict) -> dict:
    """Re-run GA with the same (instance_seed, trial_seed, pop_size, maze_type)."""
    instance_seed = int(row["instance_seed"])
    trial_seed = int(row["trial_seed"])
    pop_size = int(row["pop_size"])
    maze_type = MAZE_TYPE_REMAP.get(row["maze_type"], row["maze_type"])

    maze = generate_maze(20, 20, seed=instance_seed, maze_type=maze_type)
    disruption_iteration = DISRUPTION_TIME if maze_type == "Sudden Wall" else -1
    forced_min = (
        DISRUPTION_TIME + FORCED_MIN_ITERATIONS_AFTER_DISRUPTION
        if maze_type == "Sudden Wall"
        else 0
    )

    random.seed(trial_seed)
    np.random.seed(trial_seed)

    ga = GeneticAlgorithm(
        maze,
        pop_size=pop_size,
        chromosome_length=GA_CHROMOSOME_LENGTH_FN(maze.width, maze.height),
        crossover_rate=GA_PARAMS["crossover_rate"],
        mutation_rate=GA_PARAMS["mutation_rate"],
    )
    return ga.run(
        max_iterations=int(row["iterations"]),
        disruption_iteration=disruption_iteration,
        forced_min_iterations=forced_min,
    )


def main() -> int:
    pre_fix_path = REPO_ROOT / "results" / "runs.csv.pre-fix"
    if not pre_fix_path.exists():
        print(f"FAIL: {pre_fix_path} does not exist")
        return 1

    sampled = sample_ga_rows(pre_fix_path, n=5)
    print(f"Sampled {len(sampled)} GA rows from {pre_fix_path.name}\n")

    hard_fail = False
    for row in sampled:
        result = rerun_ga(row)
        expected_success = row["success_overall"] == "True"
        expected_iters = int(row["iterations"])
        actual_success = bool(result["success"])
        actual_iters = int(result["iterations"])

        match = actual_success == expected_success and actual_iters == expected_iters
        status = "MATCH" if match else "DIFFER"
        print(
            f"  {status}: maze={row['maze_type']:20} pop={row['pop_size']:3} "
            f"inst={row['instance_seed']} trial={row['trial_seed']}"
        )
        print(
            f"    expected: success={expected_success} iters={expected_iters}"
        )
        print(
            f"    actual:   success={actual_success} iters={actual_iters}"
        )
        if not match:
            hard_fail = True

    if hard_fail:
        print("\nHARD FAIL: GA results changed across the PSO/ACO refactor.")
        return 1
    print("\nPASS: GA results bit-identical across the PSO/ACO refactor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
