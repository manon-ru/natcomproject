import matplotlib.pyplot as plt
import numpy as np
from maze.environment import MazeEnvironment

def visualize_maze(
    maze: MazeEnvironment,
    optimal_path: list[tuple] | None = None,  # RENAMED for clarity
    best_found_path: list[tuple] | None = None, # NEW PARAMETER
    all_paths: list[list[tuple]] | None = None,
    title: str = "Maze",
    show: bool = True,
    max_dim: int = 10
) -> plt.Figure:
    """
    Render the maze with faint histories, the algorithm's best found path (red),
    and the absolute optimal ground truth (green).
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

    # Draw walls (omitted loop for brevity, same as previous functioning code)
    for y in range(maze.height + 1):
        for x in range(maze.width):
            if maze.horizontal_walls[y, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color="black", lw=line_weight)
    for y in range(maze.height):
        for x in range(maze.width + 1):
            if maze.vertical_walls[y, x]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color="black", lw=line_weight)

    # Layer 1: Faint blue cloud of ALL paths (zorder=1)
    if all_paths:
        for history_list in all_paths:
            if history_list and len(history_list) > 1:
                xs = [node[0] for node in history_list]
                ys = [node[1] for node in history_list]
                ax.plot(xs, ys, color="royalblue", lw=line_weight, alpha=0.1, zorder=1)

    # Layer 2: Best Found by Algorithm in RED (zorder=2)
    if best_found_path and len(best_found_path) > 1:
        xs = [p[0] for p in best_found_path]
        ys = [p[1] for p in best_found_path]
        # Draw slightly thicker than blue, thinner than optimal red
        ax.plot(xs, ys, color="red", lw=line_weight * 1.3, alpha=0.9, zorder=2)

    # Layer 3: Absolute Optimal Ground Truth in GREEN (zorder=3)
    # Keeping this as Layer 3 ensures ground truth is always visible on top
    if optimal_path and len(optimal_path) > 1:
        xs = [p[0] for p in optimal_path]
        ys = [p[1] for p in optimal_path]
        ax.plot(xs, ys, color="lime", lw=line_weight * 1.5, alpha=0.8, zorder=3)

    # markers, aspect, bounds, same as before...
    # (S and G markers use zorder=4 to stay on very top)
    sx, sy = maze.start
    gx, gy = maze.goal
    ax.text(sx, sy, "S", color="green", weight="bold", ha="center", va="center", 
            fontsize=font_size, zorder=4)
    ax.text(gx, gy, "G", color="darkred", weight="bold", ha="center", va="center", 
            fontsize=font_size, zorder=4)

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
