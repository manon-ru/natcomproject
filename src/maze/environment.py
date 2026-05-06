from collections import deque

import numpy as np


class MazeEnvironment:
    """
    2D maze where walls block passages between cells.

    Walls are stored as boolean arrays:
      - horizontal_walls[y, x]: wall on the top edge of cell (x, y)
      - vertical_walls[y, x]:   wall on the left edge of cell (x, y)

    Coordinate convention: (x, y) tuples; numpy arrays indexed [y, x].
    """

    def __init__(self, width: int, height: int, start: tuple, goal: tuple):
        self.width = width
        self.height = height
        self.start = start  # (x, y)
        self.goal = goal    # (x, y)

        # All walls present by default (True = wall exists)
        self.horizontal_walls = np.ones((height + 1, width), dtype=bool)
        self.vertical_walls = np.ones((height, width + 1), dtype=bool)

    def remove_wall(self, c1: tuple, c2: tuple) -> None:
        """Remove the wall between two adjacent cells."""
        x1, y1 = c1
        x2, y2 = c2
        if x1 == x2:  # Moving vertically -> remove horizontal wall
            self.horizontal_walls[max(y1, y2), x1] = False
        elif y1 == y2:  # Moving horizontally -> remove vertical wall
            self.vertical_walls[y1, max(x1, x2)] = False

    def add_wall(self, c1: tuple, c2: tuple) -> None:
        """Add a wall between two adjacent cells."""
        x1, y1 = c1
        x2, y2 = c2
        if x1 == x2:
            self.horizontal_walls[max(y1, y2), x1] = True
        elif y1 == y2:
            self.vertical_walls[y1, max(x1, x2)] = True

    def has_wall_between(self, c1: tuple, c2: tuple) -> bool:
        """Return True if there is a wall blocking passage between two adjacent cells."""
        x1, y1 = c1
        x2, y2 = c2
        if x1 == x2:
            return self.horizontal_walls[max(y1, y2), x1]
        elif y1 == y2:
            return self.vertical_walls[y1, max(x1, x2)]
        return True  # Non-adjacent cells are always blocked

    def neighbors(self, x: int, y: int) -> list[tuple]:
        """Return passable neighbors of cell (x, y)."""
        result = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if not self.has_wall_between((x, y), (nx, ny)):
                    result.append((nx, ny))
        return result

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def shortest_path(self) -> list[tuple] | None:
        """
        BFS from start to goal. Returns the list of (x, y) cells forming the
        shortest path (including start and goal), or None if unreachable.
        """
        queue: deque[tuple] = deque([(self.start, [self.start])])
        visited = {self.start}

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == self.goal:
                return path
            for neighbor in self.neighbors(x, y):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None
