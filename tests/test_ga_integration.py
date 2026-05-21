import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.ga import GeneticAlgorithm
from tests.fixtures.small_mazes import corridor_5x1, trivial_3x3


def test_ga_solves_trivial_3x3():
    random.seed(42)
    maze = trivial_3x3()
    ga = GeneticAlgorithm(maze, pop_size=10)

    result = ga.run(max_iterations=50, disruption_iteration=-1, forced_min_iterations=0)

    assert result["success"] is True
    assert result["path"][0] == (0, 0)
    assert result["path"][-1] == (2, 2)
    assert result["iterations"] <= 50


def test_ga_solves_corridor_5x1():
    random.seed(42)
    maze = corridor_5x1()
    ga = GeneticAlgorithm(maze, pop_size=10)

    result = ga.run(max_iterations=50, disruption_iteration=-1, forced_min_iterations=0)

    assert result["success"] is True
    assert result["path"][0] == (0, 0)
    assert result["path"][-1] == (4, 0)
