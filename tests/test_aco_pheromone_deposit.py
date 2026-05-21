"""RED tests for canonical Dorigo et al. 1996 Ant System pheromone deposit.

These tests lock in the Q / L rule from Dorigo et al. 1996: pheromone is
deposited only after a completed path, and only by successful ants.
"""

import pytest

from algorithms.aco import ACO
from maze.environment import MazeEnvironment
from tests.fixtures.small_mazes import corridor_5x1


def _make_aco(maze, *, num_ants=1, rho=0.0, q=2.0, initial_tau=0.8):
    return ACO(
        maze,
        num_ants=num_ants,
        evaporation_rate=rho,
        pheromone_deposit=q,
        initial_pheromone=initial_tau,
    )


def _patch_next_steps(monkeypatch, steps):
    planned = iter(steps)

    def _choose_next(*_args, **_kwargs):
        try:
            return next(planned)
        except StopIteration:
            return None

    monkeypatch.setattr(ACO, "choose_next", _choose_next)


def _branch_maze_short_dead_end():
    maze = MazeEnvironment(3, 3, (0, 1), (2, 1))
    maze.remove_wall((0, 1), (1, 1))
    maze.remove_wall((1, 1), (2, 1))
    maze.remove_wall((0, 1), (0, 2))
    return maze


def _branch_maze_long_dead_end():
    maze = MazeEnvironment(4, 4, (0, 1), (3, 1))
    maze.remove_wall((0, 1), (1, 1))
    maze.remove_wall((1, 1), (2, 1))
    maze.remove_wall((2, 1), (3, 1))
    maze.remove_wall((0, 1), (0, 2))
    maze.remove_wall((0, 2), (0, 3))
    return maze


def test_deposit_amount_is_q_over_l(monkeypatch):
    """Dorigo et al. 1996: Δτ = Q / L on each cell of the completed path."""
    maze = corridor_5x1()
    aco = _make_aco(maze, rho=0.0, q=2.0, initial_tau=0.8)
    _patch_next_steps(monkeypatch, [(1, 0), (2, 0), (3, 0), (4, 0)])

    result = aco.run(max_iterations=10)
    path = result["path"]

    expected = 0.8 + (2.0 / len(path))
    for cell in path:
        assert aco.pheromones[cell[1], cell[0]] == pytest.approx(expected, abs=1e-6)


def test_no_intra_step_deposit():
    """Dorigo et al. 1996: no deposit until a path is complete."""
    maze = MazeEnvironment(10, 1, (0, 0), (9, 0))
    for x in range(9):
        maze.remove_wall((x, 0), (x + 1, 0))

    rho = 0.1
    aco = _make_aco(maze, rho=rho, q=2.0, initial_tau=0.8)
    monkeypatch = pytest.MonkeyPatch()
    _patch_next_steps(monkeypatch, [(1, 0)])

    aco.run(max_iterations=1)

    expected = 0.8
    assert all(
        value == pytest.approx(expected, abs=1e-6)
        for value in aco.pheromones.flatten()
    )
    monkeypatch.undo()


def test_failed_ants_do_not_deposit(monkeypatch):
    """Dorigo et al. 1996: only ants that reach the goal deposit."""
    maze = _branch_maze_short_dead_end()
    aco = _make_aco(maze, num_ants=2, rho=0.0, q=2.0, initial_tau=0.8)
    _patch_next_steps(monkeypatch, [(1, 1), (0, 2), (2, 1), None])

    aco.run(max_iterations=5)

    assert aco.pheromones[2, 0] == pytest.approx(1.3, abs=1e-6)


def test_short_path_gets_more_pheromone_per_cell(monkeypatch):
    """Dorigo et al. 1996: Q / L makes shorter paths stronger per cell."""
    short_maze = corridor_5x1()
    long_maze = MazeEnvironment(10, 1, (0, 0), (9, 0))
    for x in range(9):
        long_maze.remove_wall((x, 0), (x + 1, 0))

    short_aco = _make_aco(short_maze, rho=0.0, q=2.0, initial_tau=0.8)
    long_aco = _make_aco(long_maze, rho=0.0, q=2.0, initial_tau=0.8)

    _patch_next_steps(monkeypatch, [(1, 0), (2, 0), (3, 0), (4, 0)])
    short_result = short_aco.run(max_iterations=10)

    _patch_next_steps(monkeypatch, [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0)])
    long_result = long_aco.run(max_iterations=20)

    short_delta = short_aco.pheromones[short_result["path"][1][1], short_result["path"][1][0]] - 0.8
    long_delta = long_aco.pheromones[long_result["path"][1][1], long_result["path"][1][0]] - 0.8

    assert short_delta == pytest.approx(2.0 / 10.0, abs=1e-6)
    assert long_delta == pytest.approx(2.0 / 5.0, abs=1e-6)


def test_transient_branch_cells_stay_evaporation_only(monkeypatch):
    """Dorigo et al. 1996: completed-path deposit excludes dead branches."""
    maze = _branch_maze_long_dead_end()
    aco = _make_aco(maze, num_ants=2, rho=0.5, q=2.0, initial_tau=0.005)
    _patch_next_steps(
        monkeypatch,
        [(1, 1), (0, 2), (2, 1), (0, 3), (3, 1), None],
    )

    aco.run(max_iterations=6)

    assert aco.pheromones[3, 0] == pytest.approx(2.0, abs=1e-6)
