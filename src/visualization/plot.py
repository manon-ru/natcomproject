import matplotlib.pyplot as plt
import numpy as np
from maze.environment import MazeEnvironment

def visualize_maze(
    maze: MazeEnvironment,
    optimal_path: list[tuple] | None = None,
    all_paths: list[list[tuple]] | None = None,
    alt_path: list[tuple] | None = None,
    highlight_wall: tuple | None = None,
    title: str = "Maze",
    show: bool = True,
    max_dim: int = 10,
) -> plt.Figure:
    """Render the maze with optional paths and highlighted walls.

    `optimal_path` is drawn in lime green (primary route).
    `alt_path`     is drawn in orange    (secondary route, e.g. long path
                                           after a Sudden Wall is added or
                                           the second equal route in
                                           Parallel Paths).
    `highlight_wall` is drawn in red (e.g. the Sudden Wall location).
    """
    aspect_ratio = maze.width / maze.height
    if aspect_ratio > 1:
        fig_width = max_dim
        fig_height = max_dim / aspect_ratio
    else:
        fig_height = max_dim
        fig_width = max_dim * aspect_ratio

    fig, ax = plt.subplots(figsize=(max(5, fig_width), max(5, fig_height)))

    base_scale = 10 / max(maze.width, maze.height)
    line_weight = max(1, 3 * base_scale)
    font_size = max(8, 14 * base_scale)

    for y in range(maze.height + 1):
        for x in range(maze.width):
            if maze.horizontal_walls[y, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color="black", lw=line_weight)
    for y in range(maze.height):
        for x in range(maze.width + 1):
            if maze.vertical_walls[y, x]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color="black", lw=line_weight)

    if all_paths:
        for history_list in all_paths:
            if history_list and len(history_list) > 1:
                xs = [node[0] for node in history_list]
                ys = [node[1] for node in history_list]
                ax.plot(xs, ys, color="royalblue", lw=line_weight, alpha=0.1, zorder=1)

    if alt_path and len(alt_path) > 1:
        xs = [p[0] for p in alt_path]
        ys = [p[1] for p in alt_path]
        ax.plot(xs, ys, color="orange", lw=line_weight * 1.5, alpha=0.75, zorder=1.5)

    if optimal_path and len(optimal_path) > 1:
        xs = [p[0] for p in optimal_path]
        ys = [p[1] for p in optimal_path]
        ax.plot(xs, ys, color="lime", lw=line_weight * 1.5, alpha=0.8, zorder=2)

    if highlight_wall is not None:
        (wx1, wy1), (wx2, wy2) = highlight_wall
        if wx1 == wx2:
            wy = max(wy1, wy2)
            ax.plot([wx1 - 0.5, wx1 + 0.5], [wy - 0.5, wy - 0.5],
                    color="red", lw=line_weight * 3.0, solid_capstyle="round", zorder=2.5)
        else:
            wx = max(wx1, wx2)
            ax.plot([wx - 0.5, wx - 0.5], [wy1 - 0.5, wy1 + 0.5],
                    color="red", lw=line_weight * 3.0, solid_capstyle="round", zorder=2.5)

    sx, sy = maze.start
    gx, gy = maze.goal
    ax.text(sx, sy, "S", color="green", weight="bold", ha="center", va="center",
            fontsize=font_size, zorder=3)
    ax.text(gx, gy, "G", color="darkred", weight="bold", ha="center", va="center",
            fontsize=font_size, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(maze.height - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    
    if show: plt.show()
    return fig


def visualize_entropy(
    entropy_histories: dict[str, list[float]],
    sample_interval: int = 10,
    title: str = "Population Diversity Over Time",
    show: bool = True,
) -> plt.Figure:
    """
    Plot Shannon entropy over iterations for multiple algorithms.

    Args:
        entropy_histories: Dict mapping algorithm name to its entropy_history list.
        sample_interval:   Number of iterations between each entropy sample.
        title:             Plot title.
        show:              If True, call plt.show().

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, history in entropy_histories.items():
        iterations = [i * sample_interval for i in range(len(history))]
        ax.plot(iterations, history, label=name)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title(title)
    ax.legend()

    if show:
        plt.show()

    return fig