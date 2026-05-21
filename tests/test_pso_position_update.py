import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.pso import PSO
from tests.fixtures.small_mazes import corridor_5x1, simple_choice_3x3, trivial_3x3


def make_particle(position, velocity, start_cell):
    return {
        "position": np.array(position, dtype=float),
        "velocity": np.array(velocity, dtype=float),
        "path": [tuple(start_cell)],
        "personal_best": [tuple(start_cell)],
        "visited": {tuple(start_cell)},
        "history": [tuple(start_cell)],
    }


def test_position_update_adds_velocity():
    """Position update: x(t+1) = x(t) + v(t+1), per Shi & Eberhart 1998"""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=1)
    particle = make_particle((2.0, 3.0), (1.5, -0.7), (2, 3))

    pso.update_particle(particle, [maze.goal])

    np.testing.assert_allclose(particle["position"], np.array([3.5, 2.3]))


def test_discretization_floor_plus_half():
    """Discretization: cell = (floor(x+0.5), floor(y+0.5)) — project-specific rule"""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=1)

    cases = [
        ((3.5, 2.3), (4, 2)),
        ((3.4, 2.6), (3, 3)),
        ((0.49, 0.51), (0, 1)),
    ]

    for position, expected_cell in cases:
        particle = make_particle(position, (0.0, 0.0), (0, 0))
        pso.update_particle(particle, [maze.goal])
        assert particle["path"][-1] == expected_cell


def test_wall_collision_clamps_position_and_zeros_velocity():
    """Absorptive boundary: wall hit → clamp position to last valid, zero velocity"""
    maze = corridor_5x1()
    maze.add_wall((1, 0), (2, 0))
    pso = PSO(maze, num_particles=1)
    particle = make_particle((1.0, 0.0), (1.5, 0.0), (1, 0))

    pso.update_particle(particle, [maze.goal])

    np.testing.assert_allclose(particle["velocity"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(particle["position"], np.array([1.0, 0.0]))


def test_out_of_bounds_treated_as_wall():
    """Out-of-bounds treated as wall collision per absorptive boundary rule"""
    maze = corridor_5x1()
    pso = PSO(maze, num_particles=1)
    particle = make_particle((0.0, 0.0), (-2.0, 0.0), (0, 0))

    pso.update_particle(particle, [maze.goal])

    np.testing.assert_allclose(particle["velocity"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(particle["position"], np.array([0.0, 0.0]))


def test_valid_move_updates_position_and_path():
    maze = corridor_5x1()
    maze.add_wall((0, 0), (1, 0))
    pso = PSO(maze, num_particles=1)
    particle = make_particle((1.0, 0.0), (1.5, 0.0), (1, 0))

    pso.update_particle(particle, [maze.goal])

    np.testing.assert_allclose(particle["position"], np.array([2.5, 0.0]))
    assert particle["path"][-1] == (2, 0)
