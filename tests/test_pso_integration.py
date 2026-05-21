import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.pso import PSO
from tests.fixtures.small_mazes import corridor_5x1, trivial_3x3


def test_run_completes_on_small_maze():
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=50)
    assert set(result.keys()) == {
        "success",
        "iterations",
        "path",
        "snapshot",
        "snapshot_history",
        "disruption_iteration",
        "history",
    }


def test_run_completes_with_disruption():
    maze = trivial_3x3()
    maze.dynamic_wall = ((1, 0), (1, 1))
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=100, disruption_iteration=20, forced_min_iterations=40)
    assert set(result.keys()) == {
        "success",
        "iterations",
        "path",
        "snapshot",
        "snapshot_history",
        "disruption_iteration",
        "history",
    }


def test_run_with_pop_sizes():
    expected_keys = {
        "success",
        "iterations",
        "path",
        "snapshot",
        "snapshot_history",
        "disruption_iteration",
        "history",
    }
    for pop_size in [20, 50, 150]:
        maze = corridor_5x1()
        pso = PSO(maze, num_particles=pop_size)
        result = pso.run(max_iterations=30)
        assert set(result.keys()) == expected_keys
