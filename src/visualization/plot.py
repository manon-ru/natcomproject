import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from maze.environment import MazeEnvironment

def visualize_maze(
    maze: MazeEnvironment,
    path: list[tuple] | None = None,
    all_paths: list[list[tuple]] | None = None,  # NEW PARAMETER
    title: str = "Maze",
    show: bool = True,
    max_dim: int = 10
) -> plt.Figure:
    """
    Render a wall-based MazeEnvironment with adaptive scaling for large grids.
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

    # Draw horizontal walls
    for y in range(maze.height + 1):
        for x in range(maze.width):
            if maze.horizontal_walls[y, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color="black", lw=line_weight)

    # Draw vertical walls
    for y in range(maze.height):
        for x in range(maze.width + 1):
            if maze.vertical_walls[y, x]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color="black", lw=line_weight)

    # --- NEW: Overlay ALL explored paths as a faint heatmap ---
    if all_paths:
        for p in all_paths:
            if p and len(p) > 1:
                xs = [node[0] for node in p]
                ys = [node[1] for node in p]
                # Low alpha makes overlapping paths darker
                ax.plot(xs, ys, color="royalblue", lw=line_weight, alpha=0.1, zorder=1)

    # Overlay the BEST solution path in red so it stands out
    if path and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="red", lw=line_weight * 1.5, alpha=0.9, zorder=2)

    # Start and goal markers
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
    
    if show:
        plt.show()

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
