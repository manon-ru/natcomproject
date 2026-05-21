"""
TDD integration tests for ACO — end-to-end smoke tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from algorithms.aco import ACO
from tests.fixtures.small_mazes import trivial_3x3


def test_run_completes_on_small_maze():
    """ACO can complete a run on a trivial maze without exceptions."""
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=50)
    assert isinstance(result, dict)
    assert "success" in result


def test_run_completes_with_disruption():
    """ACO.run() with disruption_iteration does not crash."""
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=100, disruption_iteration=20, forced_min_iterations=40)
    assert isinstance(result, dict)
    assert "success" in result


def test_run_with_different_pop_sizes():
    """ACO works with num_ants 20, 50, 150 without exceptions."""
    maze = trivial_3x3()
    for num_ants in [20, 50, 150]:
        aco = ACO(maze, num_ants=num_ants)
        result = aco.run(max_iterations=20)
        assert isinstance(result, dict), f"num_ants={num_ants} failed"
        required = {"success", "iterations", "path", "snapshot",
                    "snapshot_history", "disruption_iteration", "history"}
        assert required <= set(result.keys()), f"num_ants={num_ants} missing keys"
