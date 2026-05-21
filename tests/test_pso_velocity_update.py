"""
TDD RED tests for canonical PSO velocity update equation.

These tests FAIL against the current pso.py (which has no velocity state) and
define what the new implementation must satisfy.

References:
    Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
    In Proceedings of the IEEE International Conference on Evolutionary
    Computation (ICEC 1998), pages 69-73.
    https://doi.org/10.1109/ICEC.1998.699146

    The inertia-weight velocity update introduced by Shi & Eberhart 1998 (Eq. 1):
        v(t+1) = ω*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)

    where:
        ω      — inertia weight (Shi & Eberhart 1998 key contribution)
        c1, c2 — cognitive / social acceleration coefficients
        r1, r2 — uniform random scalars in [0, 1], drawn per dimension per step
        pbest  — particle's personal best position
        gbest  — global (swarm) best position
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from unittest.mock import patch

from algorithms.pso import PSO
from tests.fixtures.small_mazes import trivial_3x3


def test_velocity_update_matches_canonical_formula():
    """
    Per Shi & Eberhart 1998 Eq. 1 (inertia weight form):
        v(t+1) = ω*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)

    Numerical verification with ω=1.0, c1=0.1, c2=0.2, r1=r2=0.5:
        x=(2,3), v=(0.5,0.1), pbest=(4,5), gbest=(6,7)

        v_new_x = 1.0*0.5 + 0.1*0.5*(4-2) + 0.2*0.5*(6-2)
                = 0.5    + 0.1            + 0.4            = 1.0
        v_new_y = 1.0*0.1 + 0.1*0.5*(5-3) + 0.2*0.5*(7-3)
                = 0.1    + 0.1            + 0.4            = 0.6

    RED reason: current pso.py has no update_velocity() method.
    """
    maze = trivial_3x3()
    pso = PSO(maze, omega=1.0, c1=0.1, c2=0.2)
    particles = pso.initialize_particles()
    particle = particles[0]

    particle["position"] = np.array([2.0, 3.0])
    particle["velocity"] = np.array([0.5, 0.1])
    particle["personal_best_position"] = np.array([4.0, 5.0])
    pso.global_best_position = np.array([6.0, 7.0])

    with patch("numpy.random.uniform", return_value=0.5):
        pso.update_velocity(particle)

    np.testing.assert_array_almost_equal(
        particle["velocity"],
        np.array([1.0, 0.6]),
        decimal=6,
        err_msg=(
            "Velocity update does not match Shi & Eberhart 1998 Eq. 1: "
            "v(t+1) = ω*v(t) + c1*r1*(pbest-x) + c2*r2*(gbest-x)"
        ),
    )


def test_particle_has_velocity_attribute():
    """
    Per Shi & Eberhart 1998: each particle maintains a continuous velocity
    vector that is updated each iteration via the inertia-weight formula.
    Particles returned by initialize_particles() must carry a 'velocity' key
    holding a numpy array of shape (2,) for 2-D maze navigation.

    RED reason: current initialize_particles() returns dicts with keys
    {'path', 'personal_best', 'visited', 'history'} — no 'velocity' key.
    """
    maze = trivial_3x3()
    pso = PSO(maze)
    particles = pso.initialize_particles()

    assert len(particles) > 0, "initialize_particles() returned empty list"

    for i, particle in enumerate(particles):
        assert "velocity" in particle, (
            f"Particle {i} missing 'velocity' key "
            "(required by Shi & Eberhart 1998 inertia-weight PSO)"
        )
        assert isinstance(particle["velocity"], np.ndarray), (
            f"particle[{i}]['velocity'] must be a numpy array, "
            f"got {type(particle['velocity'])}"
        )
        assert particle["velocity"].shape == (2,), (
            f"particle[{i}]['velocity'] must have shape (2,) for 2-D maze, "
            f"got shape {particle['velocity'].shape}"
        )


def test_particle_has_continuous_position():
    """
    Per Shi & Eberhart 1998: position is a continuous real-valued vector
    x ∈ ℝ², not a discrete grid path list.  The particle moves through
    continuous space; grid-cell mapping is a separate concern.

    Particles returned by initialize_particles() must carry a 'position' key
    holding a numpy array of shape (2,).

    RED reason: current initialize_particles() has no 'position' key —
    only 'path' (a list of integer grid tuples).
    """
    maze = trivial_3x3()
    pso = PSO(maze)
    particles = pso.initialize_particles()

    assert len(particles) > 0, "initialize_particles() returned empty list"

    for i, particle in enumerate(particles):
        assert "position" in particle, (
            f"Particle {i} missing 'position' key "
            "(required by Shi & Eberhart 1998 — continuous-space PSO)"
        )
        assert isinstance(particle["position"], np.ndarray), (
            f"particle[{i}]['position'] must be a numpy array, "
            f"got {type(particle['position'])}"
        )
        assert particle["position"].shape == (2,), (
            f"particle[{i}]['position'] must have shape (2,) for 2-D maze, "
            f"got shape {particle['position'].shape}"
        )
        assert np.issubdtype(particle["position"].dtype, np.floating), (
            f"particle[{i}]['position'] must be floating-point "
            "(continuous real-valued as per Shi & Eberhart 1998)"
        )


def test_velocity_uses_np_random_not_stdlib_random():
    """
    Per canonical PSO (Shi & Eberhart 1998): the stochastic coefficients r1, r2
    are drawn from a uniform distribution.  The implementation must use
    np.random.uniform() — not Python's stdlib random module — so that the RNG
    is controllable via np.random.seed() and consistent with vectorised NumPy
    operations throughout the rest of the codebase.

    RED reason: current pso.py line 1 is `import random` and uses
    random.random() inside update_particle().  After the fix (Shi & Eberhart
    1998-compliant velocity update), `import random` must be removed entirely.
    """
    pso_source_path = (
        Path(__file__).resolve().parent.parent / "src" / "algorithms" / "pso.py"
    )
    source_text = pso_source_path.read_text()

    assert "import random" not in source_text, (
        "pso.py must not import stdlib `random`; use np.random.uniform() "
        "for r1, r2 per Shi & Eberhart 1998 convention and numpy consistency"
    )
