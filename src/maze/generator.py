"""
Maze generators for Group 27 NatComp project.

All mazes share the same convention:
  - Start: top-left corner (0, 0)
  - Goal:  bottom-right corner (W-1, H-1)

Three maze types, each designed to isolate one environmental challenge:

  U-Trap          Deception.        Normal perfect maze plus a U-shaped
                                    dead-end chamber carved near the goal.
                                    The chamber opens to the NW (start
                                    direction), so a heuristic-greedy agent
                                    heading SE enters the chamber and gets
                                    stuck against its sealed S/E walls.
                                    The real path to the goal goes around.

  Sudden Wall     Non-stationarity. Perfect DFS maze plus one extra opening
                                    that creates a shortcut. The shortcut
                                    wall is stored as maze.dynamic_wall.
                                    Initially the shortcut is open (short
                                    path is optimal). At iteration T = 100
                                    the runner adds the wall back, forcing
                                    the population onto the original (long)
                                    path.

  Parallel Paths  Multimodality.    Two corridors of exactly equal length
                                    from S to G: corridor A along the
                                    left + bottom edges, corridor B along
                                    the top + right edges. The interior
                                    is split along the diagonal into two
                                    DFS-decorated regions: one hangs off
                                    corridor B as dead-end branches, the
                                    other hangs off corridor A. Total
                                    S-to-G simple paths: exactly 2,
                                    same length.
"""
import random
from collections import deque

import numpy as np

from maze.environment import MazeEnvironment


# ---------------------------------------------------------------------------
# DFS helpers
# ---------------------------------------------------------------------------

def _carve_dfs_rect(maze, x1, x2, y1, y2):
    """Randomized DFS to carve a perfect maze inside rectangle [x1, x2) x [y1, y2)."""
    if x2 <= x1 or y2 <= y1:
        return
    start = (random.randint(x1, x2 - 1), random.randint(y1, y2 - 1))
    visited = {start}
    stack = [start]
    while stack:
        x, y = stack[-1]
        nbrs = []
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if x1 <= nx < x2 and y1 <= ny < y2 and (nx, ny) not in visited:
                nbrs.append((nx, ny))
        if nbrs:
            nx, ny = random.choice(nbrs)
            maze.remove_wall((x, y), (nx, ny))
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()


def _carve_dfs_in_cells(maze, cells):
    """Randomized DFS to carve a perfect maze over an arbitrary set of cells.

    The DFS only traverses walls inside `cells`, so walls between `cells` and
    the rest of the maze are preserved.
    """
    if not cells:
        return
    cells_set = set(cells)
    start = random.choice(tuple(cells_set))
    visited = {start}
    stack = [start]
    while stack:
        x, y = stack[-1]
        nbrs = []
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in cells_set and (nx, ny) not in visited:
                nbrs.append((nx, ny))
        if nbrs:
            nx, ny = random.choice(nbrs)
            maze.remove_wall((x, y), (nx, ny))
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()


def _bfs_distances(maze, source):
    """BFS shortest-path distances from `source` to every reachable cell."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        c = queue.popleft()
        for nb in maze.neighbors(*c):
            if nb not in dist:
                dist[nb] = dist[c] + 1
                queue.append(nb)
    return dist


# ---------------------------------------------------------------------------
# Maze 1: U-Trap (deception)
# ---------------------------------------------------------------------------

def _build_u_trap(maze, W, H):
    """Carve a perfect maze with a U-shaped dead-end chamber near the goal.

    The chamber lies just NW of the goal. It is internally hollow, sealed on
    its S, E and N sides, and connected to the rest of the maze by a single
    opening on its W side at the NW corner of the chamber. A heuristic-greedy
    agent heading SE from start runs straight into this opening, enters the
    chamber, and finds the S/E exits walled off; the real path to the goal
    wraps around the chamber's south and east sides.
    """
    # Chamber size and location. Leave 1-cell margin between chamber and the
    # E/S edges of the maze so the goal corner stays outside the chamber and
    # there is a "go-around" corridor.
    trap_w = max(4, W // 5)
    trap_h = max(4, H // 5)
    trap_x2 = W - 1            # exclusive; chamber spans columns [trap_x1, trap_x2)
    trap_y2 = H - 1            # chamber spans rows [trap_y1, trap_y2)
    trap_x1 = trap_x2 - trap_w
    trap_y1 = trap_y2 - trap_h

    # Cells inside the chamber, marked so the main DFS won't enter them.
    trap_cells = {(x, y) for x in range(trap_x1, trap_x2)
                          for y in range(trap_y1, trap_y2)}
    outside_cells = {(x, y) for x in range(W) for y in range(H)
                            if (x, y) not in trap_cells}

    # 1. Carve a perfect maze over everything outside the chamber.
    _carve_dfs_in_cells(maze, outside_cells)

    # 2. Open the chamber interior (remove all internal walls).
    for y in range(trap_y1, trap_y2):
        for x in range(trap_x1, trap_x2):
            if x + 1 < trap_x2:
                maze.remove_wall((x, y), (x + 1, y))
            if y + 1 < trap_y2:
                maze.remove_wall((x, y), (x, y + 1))

    # 3. Single entrance: the NW corner cell of the chamber connects west to
    #    the outside maze. All other chamber boundary walls remain solid, so
    #    the chamber is a dead-end pocket.
    maze.remove_wall((trap_x1, trap_y1), (trap_x1 - 1, trap_y1))


# ---------------------------------------------------------------------------
# Maze 2: Sudden Wall (non-stationarity)
# ---------------------------------------------------------------------------

def _build_sudden_wall(maze, W, H):
    """Perfect maze with one added shortcut.

    Strategy:
      1. Generate a perfect DFS maze. The unique S->G path is L_long cells.
      2. For every still-standing wall between two reachable cells, compute
         what S->G length we would get if we removed it, using BFS distances
         from S and G in the current maze.
      3. Remove the wall that yields the biggest reduction. Now there are
         two routes: the new short path (going through the removed wall) and
         the original long path.
      4. Record that wall as `maze.dynamic_wall`. At iteration T = 100 the
         runner adds it back, which collapses the short path and forces the
         population onto the long alternative.
    """
    _carve_dfs_rect(maze, 0, W, 0, H)

    d_S = _bfs_distances(maze, maze.start)
    d_G = _bfs_distances(maze, maze.goal)
    L_long = d_S[maze.goal]

    best_len = L_long
    best_wall = None
    # Iterate over each cell and its east / south neighbour. This visits every
    # wall exactly once.
    for y in range(H):
        for x in range(W):
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx >= W or ny >= H:
                    continue
                if not maze.has_wall_between((x, y), (nx, ny)):
                    continue
                a = (x, y)
                b = (nx, ny)
                # Length of shortest S->G path that uses the new edge a-b
                # exactly once, in either direction.
                cand = min(
                    d_S.get(a, float("inf")) + 1 + d_G.get(b, float("inf")),
                    d_S.get(b, float("inf")) + 1 + d_G.get(a, float("inf")),
                )
                if cand < best_len:
                    best_len = cand
                    best_wall = (a, b)

    if best_wall is None:
        # Pathological case: no shortcut exists. Leave maze as a perfect maze
        # without a dynamic wall.
        return

    maze.remove_wall(*best_wall)
    maze.dynamic_wall = best_wall


# ---------------------------------------------------------------------------
# Maze 3: Parallel Paths (multimodality)
# ---------------------------------------------------------------------------

def _build_parallel_paths(maze, W, H):
    """Two equal-length S->G corridors, with DFS-decorated interior.

    Corridor A: (0,0) -> (0,H-1) -> (W-1,H-1)   (left edge, then bottom edge)
    Corridor B: (0,0) -> (W-1,0) -> (W-1,H-1)   (top edge,  then right edge)

    Both corridors have exactly W + H - 1 cells, sharing only start and goal.

    The interior cells (1..W-2 x 1..H-2) are split along the diagonal y = x:
      - Region 1 (x >= y): dead-end branches that hang off corridor B.
      - Region 2 (x <  y): dead-end branches that hang off corridor A.

    Each region is a DFS tree connected to its corridor by a SINGLE opening,
    so the interior cannot create a third S->G route. Total S->G simple
    paths in the resulting maze: exactly two, both of length W + H - 1.
    """
    # 1. Carve corridor A (left + bottom edges).
    for y in range(H - 1):
        maze.remove_wall((0, y), (0, y + 1))
    for x in range(W - 1):
        maze.remove_wall((x, H - 1), (x + 1, H - 1))

    # 2. Carve corridor B (top + right edges).
    for x in range(W - 1):
        maze.remove_wall((x, 0), (x + 1, 0))
    for y in range(H - 1):
        maze.remove_wall((W - 1, y), (W - 1, y + 1))

    # 3. Partition interior cells into Region 1 and Region 2.
    region1 = set()  # x >= y -> branches off corridor B
    region2 = set()  # x <  y -> branches off corridor A
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if x >= y:
                region1.add((x, y))
            else:
                region2.add((x, y))

    # 4. Carve a DFS spanning tree inside each region.
    _carve_dfs_in_cells(maze, region1)
    _carve_dfs_in_cells(maze, region2)

    # 5. Connect each region to its assigned corridor with exactly one opening.
    #    We only consider walls between region cells and the corridor cells
    #    explicitly listed for that region; this guarantees no shortcut to
    #    the other corridor.
    r1_to_B = []
    for (x, y) in region1:
        if y == 1:                                    # adjacent to top-edge corridor B
            r1_to_B.append(((x, 1), (x, 0)))
        if x == W - 2:                                # adjacent to right-edge corridor B
            r1_to_B.append(((W - 2, y), (W - 1, y)))
    if r1_to_B:
        maze.remove_wall(*random.choice(r1_to_B))

    r2_to_A = []
    for (x, y) in region2:
        if x == 1:                                    # adjacent to left-edge corridor A
            r2_to_A.append(((1, y), (0, y)))
        if y == H - 2:                                # adjacent to bottom-edge corridor A
            r2_to_A.append(((x, H - 2), (x, H - 1)))
    if r2_to_A:
        maze.remove_wall(*random.choice(r2_to_A))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_maze(width: int, height: int, seed: int = 2026,
                  maze_type: str = "Random") -> MazeEnvironment:
    """Generate a maze of the requested type.

    All maze types use start = (0, 0) and goal = (W-1, H-1).
    """
    random.seed(seed)
    np.random.seed(seed)

    start = (0, 0)
    goal = (width - 1, height - 1)
    maze = MazeEnvironment(width, height, start=start, goal=goal)

    if maze_type == "Random":
        _carve_dfs_rect(maze, 0, width, 0, height)
    elif maze_type == "U-Trap":
        _build_u_trap(maze, width, height)
    elif maze_type == "Sudden Wall":
        _build_sudden_wall(maze, width, height)
    elif maze_type == "Parallel Paths":
        _build_parallel_paths(maze, width, height)
    else:
        raise ValueError(f"Unknown maze_type: {maze_type!r}")

    return maze
