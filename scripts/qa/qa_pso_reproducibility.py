#!/usr/bin/env python3
"""
QA: Verify PSO produces identical output for the same np.random seed.

Canonical PSO uses np.random exclusively (no stdlib random), so seeding
np.random must produce deterministic results.

Exit 0 on success, 1 on failure.
"""
import sys
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from maze.generator import generate_maze
from algorithms.pso import PSO
from config import PSO_PARAMS


def run_pso(seed: int, maze_type: str = "Shortest Path Trap") -> dict:
    random.seed(seed)
    np.random.seed(seed)
    maze = generate_maze(10, 10, seed=seed, maze_type=maze_type)
    random.seed(seed + 1000)
    np.random.seed(seed + 1000)
    pso = PSO(maze, num_particles=10, **PSO_PARAMS)
    return pso.run(max_iterations=100)


def main() -> int:
    seed = 42
    result_a = run_pso(seed)
    result_b = run_pso(seed)

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
        print(f"PASS: PSO is reproducible with np.random.seed({seed})")
        print(f"  success={result_a['success']}, iterations={result_a['iterations']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
