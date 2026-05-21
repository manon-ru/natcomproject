#!/usr/bin/env python3
"""
Pre-flight validation: checks all (maze_type, seed) combinations for reachability
before launching the full 2,700-run experiment.
Run: python scripts/preflight_validate_seeds.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from config import MAZE_TYPES, INSTANCE_SEEDS, MAZE_WIDTH, MAZE_HEIGHT
except ImportError:
    # Fallback if config.py not yet available
    MAZE_TYPES = ["Shortest Path Trap", "Sudden Wall", "Parallel Paths"]
    INSTANCE_SEEDS = list(range(1, 11))
    MAZE_WIDTH = 40
    MAZE_HEIGHT = 40

from maze.generator import generate_maze


def validate_all():
    failures = []
    total = len(MAZE_TYPES) * len(INSTANCE_SEEDS)
    passed = 0

    for maze_type in MAZE_TYPES:
        for seed in INSTANCE_SEEDS:
            maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT, seed=seed, maze_type=maze_type)

            pre_path = maze.shortest_path()
            if pre_path is None:
                failures.append((maze_type, seed, "PRE_UNREACHABLE"))
                print(f"FAIL {maze_type:>15s} seed={seed:>2d} PRE_UNREACHABLE")
                continue

            pre_len = len(pre_path) - 1
            post_str = "  -"

            if maze_type == "Sudden Wall" and getattr(maze, "dynamic_wall", None):
                w1, w2 = maze.dynamic_wall
                maze.add_wall(w1, w2)
                post_path = maze.shortest_path()
                maze.remove_wall(w1, w2)

                if post_path is None:
                    failures.append((maze_type, seed, "POST_UNREACHABLE"))
                    print(f"FAIL {maze_type:>15s} seed={seed:>2d} POST_UNREACHABLE pre={pre_len}")
                    continue

                post_str = str(len(post_path) - 1)

            print(f"OK   {maze_type:>15s} seed={seed:>2d} pre={pre_len:>3d} post={post_str:>3s}")
            passed += 1

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for maze_type, seed, reason in failures:
            print(f"  {maze_type} seed={seed}: {reason}")
        print(f"VALIDATION FAILED {passed}/{total}")
        sys.exit(1)
    else:
        print(f"ALL_REACHABLE {passed}/{total}")
        sys.exit(0)


if __name__ == "__main__":
    validate_all()
