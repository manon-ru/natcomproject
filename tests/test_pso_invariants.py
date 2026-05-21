"""
TDD RED tests for cross-cutting PSO swarm invariants.

These tests define properties the new canonical PSO must satisfy.
Some fail on current pso.py due to known bugs P3 and P4.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from algorithms.pso import PSO
from tests.fixtures.small_mazes import trivial_3x3


def test_no_import_random():
    """
    PSO must use np.random exclusively, not stdlib random (P4 fix).
    Per Shi & Eberhart 1998: r1, r2 drawn from uniform distribution.
    Current pso.py has `import random` at line 1 — this test catches that.
    """
    src = (Path(__file__).parent.parent / "src" / "algorithms" / "pso.py").read_text()
    assert "import random" not in src, (
        "pso.py must not import stdlib random. Use np.random exclusively. "
        "Remove `import random` to fix the P4 bug."
    )


def test_particle_has_velocity_attribute():
    """
    Per Shi & Eberhart 1998: each particle maintains a continuous velocity vector.
    Current pso.py has no 'velocity' key in particle dict — this test catches that.
    """
    maze = trivial_3x3()
    pso = PSO(maze)
    particles = pso.initialize_particles()
    for i, p in enumerate(particles):
        assert "velocity" in p, (
            f"Particle {i} missing 'velocity' key. "
            "Per Shi & Eberhart 1998, each particle must maintain a velocity vector."
        )


def test_particle_has_continuous_position():
    """
    Per Shi & Eberhart 1998: position is a continuous real-valued vector.
    Current pso.py has no 'position' key in particle dict — this test catches that.
    """
    maze = trivial_3x3()
    pso = PSO(maze)
    particles = pso.initialize_particles()
    for i, p in enumerate(particles):
        assert "position" in p, (
            f"Particle {i} missing 'position' key. "
            "Per Shi & Eberhart 1998, each particle must maintain a continuous position."
        )


def test_run_returns_required_dict_keys():
    """
    PSO.run() must return dict with all required keys per runner.py contract.
    """
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=30)
    required = {"success", "iterations", "path", "snapshot", "snapshot_history",
                "disruption_iteration", "history"}
    missing = required - set(result.keys())
    assert not missing, f"run() result missing keys: {missing}"


def test_entropy_history_populated():
    """
    PSO must populate entropy_history during run().
    """
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    pso.run(max_iterations=30)
    assert hasattr(pso, "entropy_history"), "PSO must have entropy_history attribute"
    assert len(pso.entropy_history) > 0, "entropy_history must be non-empty after run()"
