"""
TDD integration tests for PSO — end-to-end smoke tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from algorithms.pso import PSO
from tests.fixtures.small_mazes import trivial_3x3


def test_run_completes_on_small_maze():
    """PSO can complete a run on a trivial maze without exceptions."""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=50)
    assert isinstance(result, dict)
    assert "success" in result


def test_run_completes_with_disruption():
    """PSO.run() with disruption_iteration does not crash."""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=100, disruption_iteration=20, forced_min_iterations=40)
    assert isinstance(result, dict)
    assert "success" in result


def test_run_with_different_pop_sizes():
    """PSO works with pop_size 20, 50, 150 without exceptions."""
    maze = trivial_3x3()
    for pop_size in [20, 50, 150]:
        pso = PSO(maze, num_particles=pop_size)
        result = pso.run(max_iterations=20)
        assert isinstance(result, dict), f"pop_size={pop_size} failed"
        required = {"success", "iterations", "path", "snapshot",
                    "snapshot_history", "disruption_iteration", "history"}
        assert required <= set(result.keys()), f"pop_size={pop_size} missing keys"
