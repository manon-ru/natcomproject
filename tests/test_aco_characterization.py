import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from algorithms.aco import ACO
from tests.fixtures.small_mazes import trivial_3x3


def test_run_returns_required_dict_keys():
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=30)
    assert set(result.keys()) == {
        "success",
        "iterations",
        "path",
        "snapshot",
        "snapshot_history",
        "disruption_iteration",
        "history",
    }


def test_run_success_path_starts_at_start_ends_at_goal():
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=10)
    result = aco.run(max_iterations=200)
    if result["success"]:
        assert result["path"][0] == maze.start
        assert result["path"][-1] == maze.goal


def test_run_success_path_4_connected():
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=10)
    result = aco.run(max_iterations=200)
    if result["success"]:
        for current, nxt in zip(result["path"], result["path"][1:]):
            assert abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]) == 1


def test_reproducibility_same_seed_same_output():
    maze_1 = trivial_3x3()

    np.random.seed(42)
    aco_1 = ACO(maze_1, num_ants=5)
    result_1 = aco_1.run(max_iterations=50)

    np.random.seed(42)
    maze_2 = trivial_3x3()
    aco_2 = ACO(maze_2, num_ants=5)
    result_2 = aco_2.run(max_iterations=50)

    assert result_1["path"] == result_2["path"]
    assert aco_1.entropy_history == aco_2.entropy_history


def test_pheromone_matrix_accessible_after_run():
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=5)
    aco.run(max_iterations=30)
    assert isinstance(aco.pheromones, np.ndarray)
    assert aco.pheromones.shape == (maze.height, maze.width)
    assert np.all(aco.pheromones >= 0.01)
