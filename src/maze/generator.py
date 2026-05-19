"""
Maze generators for Group 27 NatComp project.

Convention shared by all maze types:
  - Start: top-left corner (0, 0)
  - Goal:  bottom-right corner (W-1, H-1)

Maze types:

  U-Trap          Deception.        Plain perfect DFS maze. Heuristic-greedy
                                    agents are attracted toward the goal but
                                    hit the maze's natural dead-ends and must
                                    backtrack. We do NOT carve a special
                                    chamber - the trap effect emerges from
                                    the natural dead-ends of the DFS maze.

  Sudden Wall     Non-stationarity. Perfect DFS maze plus one extra opening
                                    that creates a shortcut. The shortcut
                                    wall is stored as maze.dynamic_wall.
                                    Initially the shortcut is open (short
                                    path is optimal). At iteration T = 100
                                    the runner adds the wall back, forcing
                                    the population onto the original (long)
                                    path.

  Parallel Paths  Multimodality.    Two monotonic staircase routes from S
                                    to G that zigzag through the interior:
                                      Path A stays strictly above y = x.
                                      Path B stays strictly below y = x,
                                              with one small +6-cell detour
                                              placed near the goal.
                                    They share only S and G. Path A has
                                    length W+H-1; path B has length W+H+5
                                    (with the default depth-3 detour). DFS
                                    dead-end branches decorate the interior
                                    so no third S->G route exists.
"""
import random
from collections import deque

import numpy as np

from maze.environment import MazeEnvironment


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


# Maze 1: U-Trap (deception).
# A real 7-cell U-shaped corridor placed near the goal. The agent enters at
# the top of the left arm, descends, crosses the bottom, ascends the right
# arm, and hits a dead-end. The bottom-right corner of the U is the closest
# cell to the goal in Manhattan distance, but it has no exit toward G - to
# escape, the agent must back-track up the right arm and out, increasing
# its distance to G (i.e. moving against the heuristic gradient).
# The rest of the maze is a normal DFS perfect maze that does NOT touch the
# trap, so the trap looks like a natural 7-cell dead-end branch.

def _build_u_trap(maze, W, H):
    a, b = W - 5, H - 5
    u_path = [
        (a, b),              # entrance (top of left arm)
        (a, b + 1),          # left arm
        (a, b + 2),          # bottom-left corner
        (a + 1, b + 2),      # bottom middle
        (a + 2, b + 2),      # bottom-right corner (closest to G)
        (a + 2, b + 1),      # right arm
        (a + 2, b),          # dead-end (top of right arm)
    ]
    trap_cells = set(u_path)

    rest_cells = {(x, y) for y in range(H) for x in range(W)
                  if (x, y) not in trap_cells}
    _carve_dfs_in_cells(maze, rest_cells)

    for i in range(len(u_path) - 1):
        maze.remove_wall(u_path[i], u_path[i + 1])

    if b - 1 >= 0:
        maze.remove_wall(u_path[0], (a, b - 1))

    maze.trap_path = u_path


# Maze 2: Sudden Wall (non-stationarity).
# DFS perfect maze + one added shortcut. The added-back wall blocks the
# shortcut at iteration T=100, forcing the long path.

def _build_sudden_wall(maze, W, H):
    _carve_dfs_rect(maze, 0, W, 0, H)

    d_S = _bfs_distances(maze, maze.start)
    d_G = _bfs_distances(maze, maze.goal)
    L_long = d_S[maze.goal]

    best_len = L_long
    best_wall = None
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
                cand = min(
                    d_S.get(a, float("inf")) + 1 + d_G.get(b, float("inf")),
                    d_S.get(b, float("inf")) + 1 + d_G.get(a, float("inf")),
                )
                if cand < best_len:
                    best_len = cand
                    best_wall = (a, b)

    if best_wall is None:
        return

    maze.remove_wall(*best_wall)
    maze.dynamic_wall = best_wall


# Maze 3: Parallel Paths (multimodality).
# Two RANDOM non-crossing monotonic routes from S to G - different per seed.
# Path A starts with R (and ends with D); path B starts with D (and ends with
# R). At every intermediate step path B's row is strictly larger than path
# A's row, so the two paths share only S and G. Both paths have length
# 2N - 1 cells. Cells off either path are decorated with DFS dead-end
# branches, each branch attached to exactly one path - so the maze still has
# exactly two S-G routes.

def _generate_random_non_crossing_paths(N):
    interior_a = ["R"] * (N - 2) + ["D"] * (N - 2)
    random.shuffle(interior_a)
    moves_a = ["R"] + interior_a + ["D"]

    path_a = [(0, 0)]
    for m in moves_a:
        x, y = path_a[-1]
        path_a.append((x + 1, y) if m == "R" else (x, y + 1))

    path_b = [(0, 0), (0, 1)]
    bx, by = 0, 1
    r_left = N - 1
    d_left = N - 2
    total = 2 * (N - 1)
    for step in range(2, total + 1):
        ax_at, ay_at = path_a[step]
        if step == total:
            move = "R"
        else:
            can_r = r_left > 0 and by > ay_at
            can_d = d_left > 0
            if can_r and can_d:
                move = random.choice(("R", "D"))
            elif can_r:
                move = "R"
            elif can_d:
                move = "D"
            else:
                raise RuntimeError(f"Stuck generating path B at step {step}")
        if move == "R":
            bx += 1
            r_left -= 1
        else:
            by += 1
            d_left -= 1
        path_b.append((bx, by))

    return path_a, path_b


DETOUR_DEPTH = 3                    # Detour adds 2 * DETOUR_DEPTH = 6 extra cells.
DETOUR_NEAR_END_STEPS = 10          # Detour location restricted to last K interior steps of B.


def _detour_cells(path_b, i, depth, dir_sign):
    """Cells to insert between path_b[i] and path_b[i+1] for a depth-d perpendicular detour.

    For an R-step (bx, by) -> (bx+1, by), the detour goes perpendicularly in the y
    direction (down if dir_sign=+1, up if dir_sign=-1), traverses 2*depth+1 edges,
    and visits 2*depth new cells. Symmetric for a D-step (perpendicular in x).
    Returns the ordered list of 2*depth cells to splice in, or None if path_b[i] ->
    path_b[i+1] is not a unit R/D move.
    """
    bx, by = path_b[i]
    nx, ny = path_b[i + 1]
    cells = []
    if nx == bx + 1 and ny == by:
        for k in range(1, depth + 1):
            cells.append((bx, by + dir_sign * k))
        for k in range(depth, 0, -1):
            cells.append((bx + 1, by + dir_sign * k))
        return cells
    if nx == bx and ny == by + 1:
        for k in range(1, depth + 1):
            cells.append((bx + dir_sign * k, by))
        for k in range(depth, 0, -1):
            cells.append((bx + dir_sign * k, by + 1))
        return cells
    return None


def _try_insert_detour_on_b(path_a, path_b, N,
                            depth=DETOUR_DEPTH,
                            near_end_steps=DETOUR_NEAR_END_STEPS):
    """Insert a depth-d perpendicular detour on path B near the goal.

    A depth-d detour replaces a single unit step of B with a rectangular excursion
    of 2*d+1 steps, adding exactly 2*d cells to path B. With the default depth=3
    the detour visits 6 extra cells. Candidates are restricted to the last
    `near_end_steps` interior positions of path B so the detour sits near the goal,
    where the proposal's "small detour near the end of one route" places it.
    A candidate is feasible iff every detour cell is inside the grid, not on path
    A, and not on path B. If no near-end candidate fits at depth d, we try depth
    d-1 in the same near-end window. If that also fails, path B is returned
    unchanged and the caller may retry path generation.
    """
    cells_on_a = set(path_a)
    cells_on_b = set(path_b)

    def is_feasible(cells):
        if cells is None:
            return False
        for c in cells:
            if not (0 <= c[0] < N and 0 <= c[1] < N):
                return False
            if c in cells_on_a or c in cells_on_b:
                return False
        return True

    n_interior = len(path_b) - 2
    window = min(near_end_steps, n_interior)
    start_idx = len(path_b) - 1 - window      # inclusive lower bound on candidate i

    for try_depth in range(depth, 0, -1):
        candidates = []
        for i in range(start_idx, len(path_b) - 1):
            for dir_sign in (+1, -1):
                cells = _detour_cells(path_b, i, try_depth, dir_sign)
                if is_feasible(cells):
                    candidates.append((i, cells))
        if candidates:
            i, cells = random.choice(candidates)
            return path_b[: i + 1] + cells + path_b[i + 1 :]

    return path_b


def _connected_components(cells, W, H):
    """Rook-connected components within a set of cells. Returns a list of sets."""
    remaining = set(cells)
    components = []
    while remaining:
        seed = next(iter(remaining))
        comp = {seed}
        queue = deque([seed])
        remaining.remove(seed)
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _build_parallel_paths(maze, W, H):
    if W != H:
        raise ValueError("Parallel Paths currently requires a square maze (W == H).")
    N = W

    # Retry path generation until a near-end detour fits. Empirically 1-2 attempts
    # suffice for N=40 because most random shuffles leave room near (N-1, N-1).
    max_attempts = 200
    for _ in range(max_attempts):
        path_a, path_b_base = _generate_random_non_crossing_paths(N)
        path_b = _try_insert_detour_on_b(path_a, path_b_base, N)
        if len(path_b) > len(path_b_base):
            break
    assert path_a[0] == (0, 0) and path_a[-1] == (N - 1, N - 1)
    assert path_b[0] == (0, 0) and path_b[-1] == (N - 1, N - 1)

    cells_on_a = set(path_a)
    cells_on_b = set(path_b)
    assert cells_on_a & cells_on_b == {(0, 0), (N - 1, N - 1)}

    for i in range(len(path_a) - 1):
        maze.remove_wall(path_a[i], path_a[i + 1])
    for i in range(len(path_b) - 1):
        maze.remove_wall(path_b[i], path_b[i + 1])

    rest = {(x, y) for y in range(H) for x in range(W)
            if (x, y) not in cells_on_a and (x, y) not in cells_on_b}

    # Correctness invariant: each connected component of "rest" cells is
    # attached to A xor B with exactly ONE opening, so no component creates a
    # third S->G simple path. Components touching only one path go to that
    # path; channel components touching both go to a uniformly random side.
    for comp in _connected_components(rest, W, H):
        _carve_dfs_in_cells(maze, comp)
        links_a = []
        links_b = []
        for (x, y) in comp:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if not (0 <= nb[0] < W and 0 <= nb[1] < H):
                    continue
                if not maze.has_wall_between((x, y), nb):
                    continue
                if nb in cells_on_a:
                    links_a.append(((x, y), nb))
                elif nb in cells_on_b:
                    links_b.append(((x, y), nb))
        if links_a and links_b:
            chosen = links_a if random.random() < 0.5 else links_b
        elif links_a:
            chosen = links_a
        elif links_b:
            chosen = links_b
        else:
            continue
        maze.remove_wall(*random.choice(chosen))


def generate_maze(width: int, height: int, seed: int = 2026,
                  maze_type: str = "Random") -> MazeEnvironment:
    """All maze types use start = (0, 0) and goal = (W-1, H-1)."""
    random.seed(seed)
    np.random.seed(seed)

    maze = MazeEnvironment(width, height,
                           start=(0, 0), goal=(width - 1, height - 1))

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
