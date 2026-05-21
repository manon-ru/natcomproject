import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.aco import ACO
from tests.fixtures.small_mazes import corridor_5x1, trivial_3x3


def test_run_completes_on_small_maze():
    maze = corridor_5x1()
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=50, disruption_iteration=-1, forced_min_iterations=0)
    assert isinstance(result, dict)


def test_run_completes_with_disruption():
    maze = trivial_3x3()
    maze.dynamic_wall = ((1, 0), (1, 1))
    aco = ACO(maze, num_ants=5)
    result = aco.run(max_iterations=100, disruption_iteration=20, forced_min_iterations=40)
    assert isinstance(result, dict)


def test_run_with_pop_sizes():
    for pop_size in [20, 50, 150]:
        maze = trivial_3x3()
        aco = ACO(maze, num_ants=pop_size)
        result = aco.run(max_iterations=20, disruption_iteration=-1, forced_min_iterations=0)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "success",
            "iterations",
            "path",
            "snapshot",
            "snapshot_history",
            "disruption_iteration",
            "history",
        }
