"""Generate clean, title-free maze panels for the report's Methods figure.

One representative instance (seed 1) per maze type, with the engineered
structure highlighted. Explanation is left to the LaTeX caption.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maze.generator import generate_maze
from visualization.plot import visualize_maze

OUT_DIR = "figures"
SIZE, SEED = 40, 1


def save(fig, name):
    fig.savefig(f"{OUT_DIR}/{name}", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


# Shortest Path Trap: real path (green) vs trap corridor (orange).
m = generate_maze(SIZE, SIZE, seed=SEED, maze_type="Shortest Path Trap")
save(visualize_maze(m, optimal_path=m.shortest_path(), alt_path=m.trap_path,
                    title="", show=False), "maze_trap_clean.png")

# Sudden Wall: short route (green), long route after the wall (orange), wall (red).
m = generate_maze(SIZE, SIZE, seed=SEED, maze_type="Sudden Wall")
short = m.shortest_path()
w1, w2 = m.dynamic_wall
m.add_wall(w1, w2)
long = m.shortest_path()
m.remove_wall(w1, w2)
save(visualize_maze(m, optimal_path=short, alt_path=long,
                    highlight_wall=(w1, w2), title="", show=False),
     "maze_wall_clean.png")

# Parallel Paths: the two routes (green, orange).
m = generate_maze(SIZE, SIZE, seed=SEED, maze_type="Parallel Paths")
a = m.shortest_path()
from copy import deepcopy
work = deepcopy(m)
mid = len(a) // 2
work.add_wall(a[mid - 1], a[mid])
b = work.shortest_path()
save(visualize_maze(m, optimal_path=a, alt_path=b, title="", show=False),
     "maze_parallel_clean.png")

print("Done.")
