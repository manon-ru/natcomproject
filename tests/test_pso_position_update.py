"""
TDD RED tests for PSO position update rule, discretization, and wall collision.

These tests FAIL against the current pso.py (which has no continuous position or
velocity fields) and define what the new implementation must satisfy.

References:
    Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
    In Proceedings of the IEEE International Conference on Evolutionary
    Computation (ICEC 1998), pages 69-73.
    https://doi.org/10.1109/ICEC.1998.699146

    Position update (Shi & Eberhart 1998):
        x(t+1) = x(t) + v(t+1)

    Discretization rule (project-specific):
        cell = (floor(x + 0.5), floor(y + 0.5))

    Absorptive wall collision (project-specific):
        wall hit → position clamped to last valid, velocity zeroed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from algorithms.pso import PSO
from tests.fixtures.small_mazes import trivial_3x3, corridor_5x1, simple_choice_3x3


def test_position_update_adds_velocity():
    """
    Position update: x(t+1) = x(t) + v(t+1), per Shi & Eberhart 1998.

    Numerical verification:
        x=(2.0, 1.0), v=(0.5, -0.5)
        x_new = (2.0 + 0.5, 1.0 + (-0.5)) = (2.5, 0.5)

    RED reason: current pso.py has no 'position' key in particle dict
    and no update_position() method.
    """
    maze = trivial_3x3()
    pso = PSO(maze)
    particles = pso.initialize_particles()
    particle = particles[0]

    particle["position"] = np.array([2.0, 1.0])
    particle["velocity"] = np.array([0.5, -0.5])

    pso.update_position(particle)

    np.testing.assert_array_almost_equal(
        particle["position"],
        np.array([2.5, 0.5]),
        decimal=6,
        err_msg=(
            "Position update does not match Shi & Eberhart 1998: "
            "x(t+1) = x(t) + v(t+1)"
        ),
    )


def test_discretization_floor_plus_half():
    """
    Discretization: cell = (floor(x+0.5), floor(y+0.5)) — project-specific rule.

    Test cases:
        continuous pos (1.6, 0.4) → cell (2, 0)
            1.6+0.5=2.1 → floor=2;  0.4+0.5=0.9 → floor=0
        continuous pos (0.49, 1.51) → cell (0, 2)
            0.49+0.5=0.99 → floor=0;  1.51+0.5=2.01 → floor=2

    RED reason: current pso.py has no discretize_position() method.
    """
    maze = trivial_3x3()
    pso = PSO(maze)

    # Case 1: (1.6, 0.4) → cell (2, 0)
    cell = pso.discretize_position(np.array([1.6, 0.4]))
    assert cell == (2, 0), (
        f"Discretization of (1.6, 0.4) expected cell (2, 0), got {cell}. "
        "Rule: cell = (floor(x+0.5), floor(y+0.5))"
    )

    # Case 2: (0.49, 1.51) → cell (0, 2)
    cell = pso.discretize_position(np.array([0.49, 1.51]))
    assert cell == (0, 2), (
        f"Discretization of (0.49, 1.51) expected cell (0, 2), got {cell}. "
        "Rule: cell = (floor(x+0.5), floor(y+0.5))"
    )


def test_wall_collision_clamps_position_and_zeros_velocity():
    """
    Absorptive boundary: wall hit → clamp position to last valid, zero velocity.

    simple_choice_3x3 has a wall between (1,0) and (1,1).
    Particle at continuous pos=(1.0, 0.0), velocity=(0.0, 1.5):
        - Moving in +y direction from cell (1,0)
        - Adjacent cell (1,1) is walled off from (1,0) → collision
        - Expected: position stays at (1.0, 0.0), velocity becomes (0.0, 0.0)

    RED reason: current pso.py has no update_position() method and no
    wall-collision/absorptive-boundary logic for continuous positions.
    """
    maze = simple_choice_3x3()
    pso = PSO(maze)
    particles = pso.initialize_particles()
    particle = particles[0]

    particle["position"] = np.array([1.0, 0.0])
    particle["velocity"] = np.array([0.0, 1.5])

    pso.update_position(particle)

    np.testing.assert_array_almost_equal(
        particle["velocity"],
        np.array([0.0, 0.0]),
        decimal=6,
        err_msg=(
            "Wall collision must zero velocity (absorptive boundary): "
            "particle at (1,0) moving toward walled cell (1,1) should have velocity zeroed"
        ),
    )
    np.testing.assert_array_almost_equal(
        particle["position"],
        np.array([1.0, 0.0]),
        decimal=6,
        err_msg=(
            "Wall collision must clamp position: "
            "particle at (1,0) hitting wall toward (1,1) must stay at (1.0, 0.0)"
        ),
    )


def test_out_of_bounds_treated_as_wall():
    """
    Out-of-bounds positions treated as wall collision (absorptive boundary).

    Particle at pos=(0.0, 0.0), velocity=(-2.0, 0.0):
        candidate continuous pos = (-2.0, 0.0)
        discretized: (floor(-2.0+0.5), floor(0.0+0.5)) = (floor(-1.5), 0) = (-2, 0)
        cell (-2, 0) is out of bounds for any maze
        → position clamped to (0.0, 0.0), velocity zeroed

    RED reason: current pso.py has no update_position() method and no
    out-of-bounds guard for continuous positions.
    """
    maze = trivial_3x3()
    pso = PSO(maze)
    particles = pso.initialize_particles()
    particle = particles[0]

    particle["position"] = np.array([0.0, 0.0])
    particle["velocity"] = np.array([-2.0, 0.0])

    pso.update_position(particle)

    np.testing.assert_array_almost_equal(
        particle["velocity"],
        np.array([0.0, 0.0]),
        decimal=6,
        err_msg=(
            "Out-of-bounds must zero velocity (absorptive boundary): "
            "particle moving to (-2, 0) out of bounds should have velocity zeroed"
        ),
    )
    np.testing.assert_array_almost_equal(
        particle["position"],
        np.array([0.0, 0.0]),
        decimal=6,
        err_msg=(
            "Out-of-bounds must clamp position: "
            "particle moving out of bounds must stay at (0.0, 0.0)"
        ),
    )
