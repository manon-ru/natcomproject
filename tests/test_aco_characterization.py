"""
TDD characterization tests for ACO — black-box contract tests.
These verify the run() return dict schema and basic behavioral properties.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from algorithms.aco import ACO
from tests.fixtures.small_mazes import trivial_3x3


def test_run_returns_required_dict_keys():
    """ACO.run() must return dict with all keys required by runner.py."""
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=30)
    required = {"success", "iterations", "path", "snapshot", "snapshot_history",
                "disruption_iteration", "history"}
    missing = required - set(result.keys())
    assert not missing, f"run() result missing keys: {missing}"


def test_run_success_path_starts_at_start():
    """When success=True, path[0] must equal maze.start."""
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=10)
    result = aco.run(max_iterations=200)
    if result["success"]:
        assert result["path"][0] == maze.start, (
            f"path[0]={result['path'][0]} != maze.start={maze.start}"
        )


def test_run_success_path_ends_at_goal():
    """When success=True, path[-1] must equal maze.goal."""
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=10)
    result = aco.run(max_iterations=200)
    if result["success"]:
        assert result["path"][-1] == maze.goal, (
            f"path[-1]={result['path'][-1]} != maze.goal={maze.goal}"
        )


def test_reproducibility_with_np_random_seed():
    """
    ACO must produce identical output for same np.random seed.
    ACO already uses np.random exclusively, so this should PASS.
    """
    maze = trivial_3x3()

    np.random.seed(42)
    aco1 = ACO(maze, num_ants=5)
    result1 = aco1.run(max_iterations=50)

    np.random.seed(42)
    aco2 = ACO(maze, num_ants=5)
    result2 = aco2.run(max_iterations=50)

    assert result1["success"] == result2["success"], (
        "Same np.random seed must produce same success outcome."
    )
    assert result1["path"] == result2["path"], (
        "Same np.random seed must produce identical paths."
    )


def test_pheromone_matrix_accessible_after_run():
    """After run, aco.pheromones is a 2D numpy array with correct shape."""
    maze = trivial_3x3()
    aco = ACO(maze, num_ants=5)
    aco.run(max_iterations=30)
    assert hasattr(aco, "pheromones"), "ACO must expose pheromones attribute"
    assert aco.pheromones.shape == (maze.height, maze.width), (
        f"pheromones.shape={aco.pheromones.shape} != ({maze.height}, {maze.width})"
    )
    assert np.all(aco.pheromones >= 0.01), "All pheromone values must be >= 0.01 (floor)"
