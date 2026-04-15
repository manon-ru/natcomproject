import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from maze.environment import MazeEnvironment


def visualize_maze(
    maze: MazeEnvironment,
    path: list[tuple] | None = None,
    title: str = "Maze",
    show: bool = True,
) -> plt.Figure:
    """
    Render a wall-based MazeEnvironment using line segments.

    Args:
        maze:  The MazeEnvironment to draw.
        path:  Optional list of (x, y) cells to overlay as the solution path.
        title: Plot title.
        show:  If True, call plt.show(). Set to False when saving to file.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(max(6, maze.width), max(6, maze.height)))

    # Draw horizontal walls (top/bottom edges of cells)
    for y in range(maze.height + 1):
        for x in range(maze.width):
            if maze.horizontal_walls[y, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color="black", lw=2)

    # Draw vertical walls (left/right edges of cells)
    for y in range(maze.height):
        for x in range(maze.width + 1):
            if maze.vertical_walls[y, x]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color="black", lw=2)

    # Overlay solution path
    if path and len(path) > 1:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="royalblue", lw=2, alpha=0.7, zorder=2)

    # Start and goal markers
    sx, sy = maze.start
    gx, gy = maze.goal
    ax.text(sx, sy, "S", color="green", weight="bold", ha="center", va="center", fontsize=14, zorder=3)
    ax.text(gx, gy, "G", color="red", weight="bold", ha="center", va="center", fontsize=14, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, maze.width - 0.5)
    ax.set_ylim(maze.height - 0.5, -0.5)  # Flip y so (0,0) is top-left
    ax.axis("off")
    ax.set_title(title)

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
