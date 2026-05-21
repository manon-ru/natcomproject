import random

from algorithms.ga import GeneticAlgorithm
from tests.fixtures.small_mazes import simple_choice_3x3, trivial_3x3


def _make_ga(mutation_rate=1.0):
    return GeneticAlgorithm(trivial_3x3(), pop_size=5, mutation_rate=mutation_rate)


def _assert_valid_path(ga, path):
    assert path[0] == ga.maze.start
    assert len(set(path)) == len(path)
    for left, right in zip(path, path[1:]):
        assert right in ga.maze.neighbors(*left)
        assert not ga.maze.has_wall_between(left, right)


def test_mutate_extends_or_changes_path_when_possible(monkeypatch):
    ga = _make_ga(mutation_rate=1.0)
    ind = {"path": [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]}

    monkeypatch.setattr(random, "randint", lambda low, high: 5)

    mutated = ga._mutate(ind)["path"]

    assert any(cell not in ind["path"][:5] for cell in mutated)


def test_mutate_returns_valid_path():
    ga = GeneticAlgorithm(simple_choice_3x3(), pop_size=5, mutation_rate=1.0)
    ind = {"path": [(0, 0), (1, 0), (2, 0)]}

    random.seed(1)
    for _ in range(30):
        mutated = ga._mutate(ind)["path"]
        _assert_valid_path(ga, mutated)


def test_mutate_unchanged_when_rng_above_rate():
    ga = _make_ga(mutation_rate=0.0)
    ind = {"path": [(0, 0), (0, 1), (0, 2)]}

    random.seed(2)
    mutated = ga._mutate(ind)

    assert mutated is ind


def test_mutate_handles_short_path_gracefully():
    ga = _make_ga(mutation_rate=1.0)
    ind = {"path": [(0, 0), (1, 0)]}

    random.seed(3)
    mutated = ga._mutate(ind)

    assert mutated is ind
