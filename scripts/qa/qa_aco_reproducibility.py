#!/usr/bin/env python3
"""
QA: Verify ACO produces identical output for the same np.random seed.

ACO uses np.random exclusively, so seeding np.random must produce
deterministic results.

Exit 0 on success, 1 on failure.
"""
import sys
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from maze.generator import generate_maze
from algorithms.aco import ACO
from config import ACO_PARAMS


def run_aco(seed: int, maze_type: str = "Shortest Path Trap") -> dict:
    random.seed(seed)
    np.random.seed(seed)
    maze = generate_maze(10, 10, seed=seed, maze_type=maze_type)
    random.seed(seed + 1000)
    np.random.seed(seed + 1000)
    aco = ACO(
        maze,
        num_ants=10,
        alpha=ACO_PARAMS["alpha"],
        beta=ACO_PARAMS["beta"],
        pheromone_deposit=ACO_PARAMS["Q"],
        evaporation_rate=ACO_PARAMS["rho"],
        initial_pheromone=ACO_PARAMS["tau0"],
    )
    return aco.run(max_iterations=100)


def main() -> int:
    seed = 42
    result_a = run_aco(seed)
    result_b = run_aco(seed)

    ok = True

    if result_a["success"] != result_b["success"]:
        print(f"FAIL: success differs: {result_a['success']} vs {result_b['success']}")
        ok = False

    if result_a["iterations"] != result_b["iterations"]:
        print(f"FAIL: iterations differ: {result_a['iterations']} vs {result_b['iterations']}")
        ok = False

    if result_a["path"] != result_b["path"]:
        print(f"FAIL: paths differ")
        ok = False

    if ok:
        print(f"PASS: ACO is reproducible with np.random.seed({seed})")
        print(f"  success={result_a['success']}, iterations={result_a['iterations']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
