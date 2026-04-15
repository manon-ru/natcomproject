import random

import numpy as np

from maze.environment import MazeEnvironment


def generate_maze(width: int, height: int, seed: int = 2026) -> MazeEnvironment:
    """
    Generate a perfect maze (every cell reachable, no loops) using a
    randomized depth-first search (recursive backtracker).

    Start is always (0, 0); goal is always (width-1, height-1).
    The seed guarantees reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)

    maze = MazeEnvironment(width, height, start=(0, 0), goal=(width - 1, height - 1))

    visited = {(0, 0)}
    stack = [(0, 0)]

    while stack:
        x, y = stack[-1]
        # Unvisited orthogonal neighbors within bounds
        candidates = [
            (x + dx, y + dy)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
            if 0 <= x + dx < width
            and 0 <= y + dy < height
            and (x + dx, y + dy) not in visited
        ]

        if candidates:
            nx, ny = random.choice(candidates)
            maze.remove_wall((x, y), (nx, ny))
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

    return maze
