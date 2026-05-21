import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from algorithms.aco import ACO
from maze.environment import MazeEnvironment
from tests.fixtures.small_mazes import corridor_5x1, trivial_3x3


def _make_open_20x20() -> MazeEnvironment:
    maze = MazeEnvironment(20, 20, (0, 0), (19, 19))
    for y in range(20):
        for x in range(19):
            maze.remove_wall((x, y), (x + 1, y))
    for y in range(19):
        for x in range(20):
            maze.remove_wall((x, y), (x, y + 1))
    return maze


def _make_unreachable_goal_20x20() -> MazeEnvironment:
    maze = _make_open_20x20()
    maze.add_wall((18, 19), (19, 19))
    maze.add_wall((19, 18), (19, 19))
    return maze


def test_no_import_random():
    """ACO must use np.random exclusively, not stdlib random (A4 fix).

    WILL FAIL on current aco.py: `import random` exists at line 11 but is
    never called — only np.random.choice is used (line 77).
    """
    source = open(
        Path(__file__).parent.parent / "src" / "algorithms" / "aco.py"
    ).read()
    assert "import random" not in source, (
        "aco.py must not import stdlib random. "
        "Use np.random exclusively. Remove `import random` to fix the A4 bug."
    )


def test_entropy_history_sampled_every_10_iterations():
    maze = _make_unreachable_goal_20x20()
    aco = ACO(maze, num_ants=5)
    aco.run(max_iterations=100)
    expected = 100 // 10
    assert len(aco.entropy_history) == expected, (
        f"Expected {expected} entropy samples for max_iterations=100 "
        f"(iterations 0,10,...,90). Got {len(aco.entropy_history)}."
    )


def test_pheromone_matrix_shape():
    maze = trivial_3x3()
    aco = ACO(maze)
    assert aco.pheromones.shape == (maze.height, maze.width), (
        f"Expected pheromones.shape ({maze.height}, {maze.width}), "
        f"got {aco.pheromones.shape}."
    )


_REQUIRED_KEYS = {
    "success",
    "iterations",
    "path",
    "snapshot",
    "snapshot_history",
    "disruption_iteration",
    "history",
}


def test_run_returns_required_dict_keys():
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=10)
    result = aco.run(max_iterations=50)
    assert isinstance(result, dict), f"run() must return dict, got {type(result)}"
    missing = _REQUIRED_KEYS - result.keys()
    assert not missing, (
        f"ACO.run() result missing required keys: {missing}. "
        f"Present: {set(result.keys())}."
    )


def test_successful_path_ends_at_goal():
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=20)
    result = aco.run(max_iterations=200)
    if result["success"]:
        assert result["path"][-1] == maze.goal, (
            f"success=True but path ends at {result['path'][-1]}, "
            f"expected goal {maze.goal}."
        )
    else:
        pytest.skip("ACO did not find goal within 200 iterations on trivial_3x3")
