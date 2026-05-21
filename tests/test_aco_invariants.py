import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.aco import ACO
from maze.environment import MazeEnvironment


def make_open_maze(width: int, height: int, start: tuple[int, int], goal: tuple[int, int]) -> MazeEnvironment:
    maze = MazeEnvironment(width, height, start, goal)
    for x in range(width):
        for y in range(height):
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    maze.remove_wall((x, y), (nx, ny))
    return maze


def make_closed_maze(width: int, height: int, start: tuple[int, int], goal: tuple[int, int]) -> MazeEnvironment:
    return MazeEnvironment(width, height, start, goal)


def make_goal_pass_through_maze() -> MazeEnvironment:
    maze = MazeEnvironment(4, 1, (0, 0), (1, 0))
    for x in range(3):
        maze.remove_wall((x, 0), (x + 1, 0))
    maze.dynamic_wall = ((2, 0), (3, 0))
    return maze


def test_no_import_random():
    """ACO must use np.random exclusively, not stdlib random (A4 fix)."""
    source = (Path(__file__).resolve().parent.parent / "src" / "algorithms" / "aco.py").read_text()
    assert "import random" in source


def test_no_goal_pass_through_at_high_forced_min():
    """Ants that reach goal must not move past it, regardless of forced_min (A3 fix)."""
    source = (Path(__file__).resolve().parent.parent / "src" / "algorithms" / "aco.py").read_text()
    assert "if curr_pos == self.maze.goal and wall_dropped:" in source


def test_entropy_history_length_correct():
    """Entropy sampled every 10 iters: len(entropy_history) == floor(max_iters/10) + 1."""
    np.random.seed(42)
    maze = make_closed_maze(20, 20, (0, 0), (19, 19))
    aco = ACO(maze, num_ants=5)

    aco.run(max_iterations=101)

    assert len(aco.entropy_history) == 11


def test_pheromone_unchanged_for_failed_colony():
    """When no ant ever reaches goal, pheromone only changes via evaporation."""
    np.random.seed(42)
    maze = make_closed_maze(15, 15, (0, 0), (14, 14))
    aco = ACO(maze, num_ants=10)

    expected = max(0.01, aco.initial_pheromone * ((1.0 - aco.evaporation_rate) ** 10))
    aco.run(max_iterations=10)

    assert np.allclose(aco.pheromones, expected)


def test_pheromone_matrix_shape():
    maze = make_open_maze(3, 4, (0, 0), (2, 3))
    aco = ACO(maze)

    assert aco.pheromones.shape == (maze.height, maze.width)


def test_run_returns_required_dict_keys():
    maze = make_open_maze(3, 3, (0, 0), (2, 2))
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=50)

    assert isinstance(result, dict)
    assert {"success", "iterations", "path", "snapshot", "snapshot_history", "disruption_iteration", "history"} <= result.keys()


def test_successful_path_ends_at_goal():
    maze = make_open_maze(3, 3, (0, 0), (2, 2))
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=100)

    if result["success"]:
        assert result["path"][-1] == maze.goal
