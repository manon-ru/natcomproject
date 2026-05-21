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
    """RED test: current code truncates at stuck point and returns without extension.
    After fix (retry-3 logic), second call picks index 1 which allows extension.

    Path: [(0,0),(1,0),(1,1),(0,1),(0,0),(1,0)] — has revisits intentionally so that
    truncating at index 5 gives path ending at (0,0) whose only unvisited-free neighbors
    are all already in the truncated set {(0,0),(1,0),(1,1),(0,1)}.
    Truncating at index 1 gives [(0,0)] whose neighbors (1,0),(0,1) are unvisited → extends.
    """
    ga = _make_ga(mutation_rate=1.0)
    ind = {"path": [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]}

    # First call returns 5 (stuck truncation point); subsequent calls return 1 (working point).
    # Current code: calls randint once → stuck → returns truncated path → test FAILS (RED).
    # Fixed code: calls randint again → gets 1 → extends → test PASSES (GREEN).
    call_count = [0]

    def mock_randint(low, high):
        call_count[0] += 1
        return 5 if call_count[0] == 1 else 1

    monkeypatch.setattr(random, "randint", mock_randint)

    mutated = ga._mutate(ind)["path"]

    # After fix: mutated path starts from [(0,0)] and extends into new cells not in the
    # original first-5 set {(0,0),(1,0),(1,1),(0,1)}.
    assert any(cell not in set(ind["path"][:5]) for cell in mutated)


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
