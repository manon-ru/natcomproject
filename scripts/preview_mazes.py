"""Generate preview PNGs of all maze types at 40x40 across several seeds.

For each maze type we render the most informative view:

  Shortest Path Trap
                  Shortest S->G path in lime green.
  Sudden Wall     Short path (green) before the wall drops, long path
                  (orange) after the wall drops, and the dynamic wall
                  location highlighted in red.
  Parallel Paths  Both equal-length routes shown together: path A in
                  lime green, path B (with detour) in orange.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maze.generator import generate_maze
from visualization.plot import visualize_maze


OUT_DIR = "figures/maze_previews"
os.makedirs(OUT_DIR, exist_ok=True)

SIZE = 40
SEEDS = [1, 2, 3, 4, 5]
MAZE_TYPES = ["Shortest Path Trap", "Sudden Wall", "Parallel Paths"]


def _paths_for_parallel(maze):
    """Return both equal-length S->G routes by BFS-blocking the first one."""
    from copy import deepcopy
    path_a = maze.shortest_path()
    if path_a is None:
        return None, None
    work = deepcopy(maze)
    if len(path_a) > 3:
        mid = len(path_a) // 2
        work.add_wall(path_a[mid - 1], path_a[mid])
    path_b = work.shortest_path()
    return path_a, path_b


for maze_type in MAZE_TYPES:
    for seed in SEEDS:
        maze = generate_maze(SIZE, SIZE, seed=seed, maze_type=maze_type)
        slug = maze_type.lower().replace(" ", "_").replace("-", "_")
        fname = f"{OUT_DIR}/{slug}_{SIZE}x{SIZE}_seed{seed}.png"

        if maze_type == "Sudden Wall" and getattr(maze, "dynamic_wall", None):
            short_path = maze.shortest_path()
            w1, w2 = maze.dynamic_wall
            maze.add_wall(w1, w2)
            long_path = maze.shortest_path()
            maze.remove_wall(w1, w2)
            short_steps = (len(short_path) - 1) if short_path else None
            long_steps = (len(long_path) - 1) if long_path else None
            title = (
                f"Sudden Wall {SIZE}x{SIZE}  seed={seed}   "
                f"short (green) = {short_steps} steps   |   "
                f"long after wall drops (orange) = {long_steps} steps   |   "
                f"dynamic wall = red"
            )
            fig = visualize_maze(
                maze,
                optimal_path=short_path,
                alt_path=long_path,
                highlight_wall=(w1, w2),
                title=title,
                show=False,
            )
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  {fname}  short={short_steps}  long={long_steps}")

        elif maze_type == "Parallel Paths":
            path_a, path_b = _paths_for_parallel(maze)
            la = (len(path_a) - 1) if path_a else None
            lb = (len(path_b) - 1) if path_b else None
            detour_cells = (lb - la) if (la is not None and lb is not None) else None
            title = (
                f"Parallel Paths {SIZE}x{SIZE}  seed={seed}   "
                f"path A (green) = {la} steps   |   "
                f"path B with +{detour_cells} detour (orange) = {lb} steps   |   "
                f"random non-crossing routes"
            )
            fig = visualize_maze(
                maze,
                optimal_path=path_a,
                alt_path=path_b,
                title=title,
                show=False,
            )
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  {fname}  A={la}  B={lb}")

        elif maze_type == "Shortest Path Trap" and getattr(maze, "trap_path", None):
            optimal = maze.shortest_path()
            opt_len = (len(optimal) - 1) if optimal else None
            trap = maze.trap_path
            trap_steps = len(trap) - 1
            title = (
                f"Shortest Path Trap {SIZE}x{SIZE}  seed={seed}   "
                f"real path R (green) = {opt_len} steps   |   "
                f"trap corridor (orange) = {trap_steps} steps, dead-ends 1 cell from G"
            )
            fig = visualize_maze(
                maze,
                optimal_path=optimal,
                alt_path=trap,
                title=title,
                show=False,
            )
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  {fname}  optimal={opt_len}  trap_cells={len(trap)}")

        else:
            optimal = maze.shortest_path()
            opt_len = (len(optimal) - 1) if optimal else None
            fig = visualize_maze(
                maze,
                optimal_path=optimal,
                title=f"{maze_type} {SIZE}x{SIZE}  seed={seed}  (optimal path: {opt_len} steps)",
                show=False,
            )
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)
            print(f"  {fname}  optimal={opt_len}")

print(f"\nDone. Previews in {OUT_DIR}/")
