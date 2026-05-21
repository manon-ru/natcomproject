import random

from algorithms.ga import GeneticAlgorithm
from tests.fixtures.small_mazes import trivial_3x3


def _make_ga():
    return GeneticAlgorithm(trivial_3x3(), pop_size=5, crossover_rate=1.0)


def _make_no_common_parents():
    parent_a = {"path": [(0, 0), (0, 1), (0, 2)]}
    parent_b = {"path": [(0, 0), (1, 0), (2, 0)]}
    return parent_a, parent_b


def _assert_valid_path(ga, path):
    assert path[0] == ga.maze.start
    for left, right in zip(path, path[1:]):
        assert right in ga.maze.neighbors(*left)
        assert not ga.maze.has_wall_between(left, right)


def test_crossover_with_no_common_cells_does_not_just_clone_parent_a():
    ga = _make_ga()
    parent_a, parent_b = _make_no_common_parents()

    random.seed(0)
    children = [ga._crossover(parent_a, parent_b)["path"] for _ in range(30)]

    assert any(child != parent_a["path"] for child in children)


def test_crossover_no_common_cells_child_is_still_valid_path():
    ga = _make_ga()
    parent_a, parent_b = _make_no_common_parents()

    random.seed(0)
    for _ in range(30):
        child = ga._crossover(parent_a, parent_b)["path"]
        _assert_valid_path(ga, child)


def test_crossover_with_common_cells_unchanged_behavior():
    ga = _make_ga()
    parent_a = {"path": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]}
    parent_b = {"path": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]}

    random.seed(42)
    child = ga._crossover(parent_a, parent_b)["path"]

    _assert_valid_path(ga, child)
