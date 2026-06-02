# Scratch tool, not for paper figures. The paper figures come from scripts/figures.py.
"""
Visual preview of the project's maze environments and analytics.

Shows all three maze tiers with their BFS-optimal paths, plus placeholder
charts illustrating what the final analytics will look like once the
algorithms are implemented.

Usage:
    uv run src/visualization/demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..","src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from maze.generator import generate_maze
from maze.environment import MazeEnvironment


# ---------------------------------------------------------------------------
# Maze tiers
# ---------------------------------------------------------------------------
TIERS = [
    {"name": "Maze 1: Shortest Path Trap", "width": 20, "height": 20, "seed": 2026, "maze_type": "Shortest Path Trap"},
    {"name": "Maze 2: Sudden Wall", "width": 20, "height": 20, "seed": 2026, "maze_type": "Sudden Wall"},
    {"name": "Maze 3: Parallel Paths", "width": 20, "height": 20, "seed": 2026, "maze_type": "Parallel Paths"},
]

ALGO_NAMES = ["GA", "PSO", "ACO"]
ALGO_COLORS = ["#3a86ff", "#8ecae6", "#57cc99"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_maze(ax: plt.Axes, maze: MazeEnvironment, path: list[tuple] | None = None) -> None:
    """Draw walls, optional path, and S/G markers onto an existing Axes."""
    # Cell backgrounds
    ax.set_facecolor("#f8f5f0")

    # Walls
    lw = max(1.0, 4.0 - maze.width * 0.1)  # Thinner lines for larger mazes
    for y in range(maze.height + 1):
        for x in range(maze.width):
            if maze.horizontal_walls[y, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5],
                        color="#1a1a2e", lw=lw, solid_capstyle="round")

    for y in range(maze.height):
        for x in range(maze.width + 1):
            if maze.vertical_walls[y, x]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5],
                        color="#1a1a2e", lw=lw, solid_capstyle="round")

    # Solution path
    if path and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="#3a86ff", lw=max(1.5, 3.0 - maze.width * 0.08),
                alpha=0.85, zorder=2, solid_capstyle="round")
        # Dots on each path cell
        ax.scatter(xs[1:-1], ys[1:-1], color="#3a86ff", s=max(4, 30 - maze.width),
                   zorder=3, alpha=0.6)

    # Start / goal
    font_size = max(7, 16 - maze.width * 0.4)
    sx, sy = maze.start
    gx, gy = maze.goal
    ax.text(sx, sy, "S", color="#2dc653", weight="bold",
            ha="center", va="center", fontsize=font_size, zorder=4)
    ax.text(gx, gy, "G", color="#e63946", weight="bold",
            ha="center", va="center", fontsize=font_size, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(maze.height - 0.5, -0.5)
    ax.axis("off")


def _dummy_entropy_curves(n_points: int = 50) -> dict[str, np.ndarray]:
    """Synthetic entropy curves — illustrative shapes only."""
    t = np.linspace(0, 1, n_points)
    return {
        "GA":          0.2 + 0.8 * np.exp(-2.5 * t) + np.random.default_rng(1).normal(0, 0.02, n_points),
        "PSO":         0.1 + 0.9 * np.exp(-5.0 * t ** 0.5) + np.random.default_rng(2).normal(0, 0.02, n_points),
        "ACO":         0.4 + 0.6 * (1 - t) ** 1.8 + np.random.default_rng(3).normal(0, 0.02, n_points),
    }


# ---------------------------------------------------------------------------
# Figure 1: Maze gallery
# ---------------------------------------------------------------------------

def _figure_maze_gallery(mazes: list[tuple]) -> plt.Figure:
    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor("#fafafa")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.08,
                           left=0.02, right=0.98, top=0.88, bottom=0.02)

    for col, (tier, maze, path) in enumerate(mazes):
        ax = fig.add_subplot(gs[0, col])
        _draw_maze(ax, maze, path)

        path_len = len(path) if path else 0
        ax.set_title(
            f"{tier['name']}  —  {tier['width']}×{tier['height']}  |  "
            f"BFS path: {path_len} cells",
            fontsize=12, fontweight="bold", pad=10, color="#1a1a2e",
        )

    fig.suptitle("Maze Environments — Group 27", fontsize=15,
                 fontweight="bold", color="#1a1a2e", y=0.97)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#3a86ff", lw=2, label="BFS shortest path"),
        Line2D([0], [0], marker="$S$", color="w", markerfacecolor="#2dc653",
               markersize=10, label="Start"),
        Line2D([0], [0], marker="$G$", color="w", markerfacecolor="#e63946",
               markersize=10, label="Goal"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               framealpha=0.9, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Analytics preview
# ---------------------------------------------------------------------------

def _figure_analytics(mazes: list[tuple]) -> plt.Figure:
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#fafafa")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.08, right=0.97, top=0.91, bottom=0.08)

    stub_note = "implement algorithms\nto populate"
    stub_color = "#cccccc"
    stub_text_color = "#999999"

    # ---- Top-left: Entropy over time ----------------------------------------
    ax_ent = fig.add_subplot(gs[0, 0])
    curves = _dummy_entropy_curves()
    iterations = np.linspace(0, 1000, 50)
    for (name, curve), color in zip(curves.items(), ALGO_COLORS):
        ax_ent.plot(iterations, np.clip(curve, 0, None), label=name,
                    color=color, lw=2)
    ax_ent.set_xlabel("Iteration", fontsize=9)
    ax_ent.set_ylabel("Shannon Entropy (bits)", fontsize=9)
    ax_ent.set_title("Population Diversity Over Time", fontsize=11, fontweight="bold")
    ax_ent.legend(fontsize=8, framealpha=0.7)
    ax_ent.set_ylim(0, 1.2)
    ax_ent.spines[["top", "right"]].set_visible(False)
    ax_ent.text(0.97, 0.95, "synthetic — illustrative only",
                transform=ax_ent.transAxes, fontsize=7, color=stub_text_color,
                ha="right", va="top", style="italic")

    # ---- Top-right: Success rate ---------------------------------------------
    ax_sr = fig.add_subplot(gs[0, 1])
    bars = ax_sr.bar(ALGO_NAMES, [0] * 4, color=ALGO_COLORS, edgecolor="white",
                     linewidth=1.2, zorder=2)
    ax_sr.set_ylim(0, 1.05)
    ax_sr.set_ylabel("Success Rate", fontsize=9)
    ax_sr.set_title("Success Rate (50 trials)", fontsize=11, fontweight="bold")
    ax_sr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_sr.spines[["top", "right"]].set_visible(False)
    ax_sr.tick_params(axis="x", labelsize=8)
    ax_sr.text(0.5, 0.5, stub_note, transform=ax_sr.transAxes,
               ha="center", va="center", fontsize=10, color=stub_text_color,
               style="italic")

    # ---- Bottom-left: Mean iteration count ----------------------------------
    ax_it = fig.add_subplot(gs[1, 0])
    ax_it.bar(ALGO_NAMES, [0] * 4, color=ALGO_COLORS, edgecolor="white",
              linewidth=1.2, zorder=2)
    ax_it.set_ylim(0, 1000)
    ax_it.set_ylabel("Mean Iterations", fontsize=9)
    ax_it.set_title("Convergence Speed", fontsize=11, fontweight="bold")
    ax_it.spines[["top", "right"]].set_visible(False)
    ax_it.tick_params(axis="x", labelsize=8)
    ax_it.text(0.5, 0.5, stub_note, transform=ax_it.transAxes,
               ha="center", va="center", fontsize=10, color=stub_text_color,
               style="italic")

    # ---- Bottom-right: Path optimality --------------------------------------
    ax_po = fig.add_subplot(gs[1, 1])
    # Show optimal reference line and BFS path lengths as real data
    _, maze_simple, path_simple = mazes[0]
    _, maze_medium, path_medium = mazes[1]
    _, maze_complex, path_complex = mazes[2]

    tier_labels = ["Simple\n(5×5)", "Medium\n(10×10)", "Complex\n(20×20)"]
    bfs_lengths = [len(p) if p else 0 for _, _, p in mazes]
    optimal_lengths = [len(p) if p else 0 for _, _, p in mazes]

    x = np.arange(len(tier_labels))
    ax_po.bar(x, bfs_lengths, color="#3a86ff", alpha=0.7, label="BFS optimal", zorder=2)
    ax_po.set_xticks(x)
    ax_po.set_xticklabels(tier_labels, fontsize=8)
    ax_po.set_ylabel("Path Length (cells)", fontsize=9)
    ax_po.set_title("Optimal Path Lengths (BFS)", fontsize=11, fontweight="bold")
    ax_po.spines[["top", "right"]].set_visible(False)
    ax_po.legend(fontsize=8, framealpha=0.7)

    # Annotate bar values
    for xi, v in zip(x, bfs_lengths):
        ax_po.text(xi, v + 0.3, str(v), ha="center", va="bottom", fontsize=9,
                   fontweight="bold", color="#1a1a2e")

    fig.suptitle("Analytics Preview — Group 27", fontsize=15,
                 fontweight="bold", color="#1a1a2e")

    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def show_demo() -> None:
    """Generate all three maze tiers and show the visualization demo."""
    mazes = []
    for tier in TIERS:
        # Extract the maze_type, defaulting to "Random" if not found
        m_type = tier.get("maze_type", "Random")
        maze = generate_maze(tier["width"], tier["height"], seed=tier["seed"], maze_type=m_type)
        path = maze.shortest_path()
        mazes.append((tier, maze, path))

    fig1 = _figure_maze_gallery(mazes)
    fig2 = _figure_analytics(mazes)

    plt.show()


if __name__ == "__main__":
    show_demo()
