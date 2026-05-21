"""
TDD characterization tests for PSO — black-box contract tests.
These verify the run() return dict schema and basic behavioral properties.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from algorithms.pso import PSO
from tests.fixtures.small_mazes import trivial_3x3


def test_run_returns_required_dict_keys():
    """PSO.run() must return dict with all keys required by runner.py."""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=30)
    required = {"success", "iterations", "path", "snapshot", "snapshot_history",
                "disruption_iteration", "history"}
    missing = required - set(result.keys())
    assert not missing, f"run() result missing keys: {missing}"


def test_run_success_path_starts_at_start():
    """When success=True, path[0] must equal maze.start."""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=10)
    result = pso.run(max_iterations=200)
    if result["success"]:
        assert result["path"][0] == maze.start, (
            f"path[0]={result['path'][0]} != maze.start={maze.start}"
        )


def test_run_success_path_ends_at_goal():
    """When success=True, path[-1] must equal maze.goal."""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=10)
    result = pso.run(max_iterations=200)
    if result["success"]:
        assert result["path"][-1] == maze.goal, (
            f"path[-1]={result['path'][-1]} != maze.goal={maze.goal}"
        )


def test_reproducibility_with_np_random_seed():
    """
    PSO must produce identical output for same np.random seed.
    Per canonical PSO: uses np.random exclusively (after P4 fix).
    NOTE: This test FAILS on current pso.py because it uses stdlib random.random()
    which is NOT controlled by np.random.seed().
    """
    maze = trivial_3x3()

    np.random.seed(42)
    pso1 = PSO(maze, num_particles=5)
    result1 = pso1.run(max_iterations=50)

    np.random.seed(42)
    pso2 = PSO(maze, num_particles=5)
    result2 = pso2.run(max_iterations=50)

    assert result1["success"] == result2["success"], (
        "Same np.random seed must produce same success outcome. "
        "Current pso.py uses stdlib random which is not seeded by np.random.seed()."
    )
    assert result1["path"] == result2["path"], (
        "Same np.random seed must produce identical paths. "
        "Current pso.py uses stdlib random which is not seeded by np.random.seed()."
    )
