import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.pso import PSO
from tests.fixtures.small_mazes import corridor_5x1, trivial_3x3


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

    pso.update_particle(particle, np.array([maze.goal]))  # type: ignore[arg-type]

    np.testing.assert_allclose(particle["position"], np.array([3.5, 2.3]))


def test_discretization_uses_round():
    """Discretization uses round() (nearest integer), not floor()."""
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=1)

    assert pso.discretize_position(np.array([3.5, 2.3])) == (4, 2)   # round(3.5)=4
    assert pso.discretize_position(np.array([3.4, 2.6])) == (3, 3)   # round(2.6)=3
    assert pso.discretize_position(np.array([0.49, 0.51])) == (0, 1) # round(0.49)=0, round(0.51)=1


def test_wall_collision_projects_to_open_neighbour():
    """Wall hit → DFS projects particle to best open unvisited neighbour."""
    maze = corridor_5x1()
    maze.add_wall((1, 0), (2, 0))
    pso = PSO(maze, num_particles=1)
    particle = make_particle((1.0, 0.0), (1.5, 0.0), (1, 0))

    np.random.seed(0)
    pso.update_particle(particle, np.array([maze.goal]))  # type: ignore[arg-type]

    # (0,0) is the only open unvisited neighbour of (1,0) after wall is added
    np.testing.assert_allclose(particle["position"], np.array([0.0, 0.0]))
    assert particle["path"][-1] == (0, 0)


def test_out_of_bounds_projects_to_open_neighbour():
    """OOB move → DFS projects particle to open unvisited neighbour."""
    maze = corridor_5x1()
    pso = PSO(maze, num_particles=1)
    particle = make_particle((0.0, 0.0), (-2.0, 0.0), (0, 0))

    np.random.seed(0)
    pso.update_particle(particle, np.array([maze.goal]))  # type: ignore[arg-type]

    # (1,0) is the only open unvisited neighbour of (0,0)
    np.testing.assert_allclose(particle["position"], np.array([1.0, 0.0]))
    assert particle["path"][-1] == (1, 0)


def test_valid_move_updates_position_and_path():
    maze = corridor_5x1()
    maze.add_wall((0, 0), (1, 0))
    pso = PSO(maze, num_particles=1)
    particle = make_particle((1.0, 0.0), (1.5, 0.0), (1, 0))

    pso.update_particle(particle, np.array([maze.goal]))  # type: ignore[arg-type]

    # Valid canonical move to (2,0) — position is snapped to cell centre
    np.testing.assert_allclose(particle["position"], np.array([2.0, 0.0]))
    assert particle["path"][-1] == (2, 0)
