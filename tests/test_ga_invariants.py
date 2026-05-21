import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.ga import GeneticAlgorithm
from maze.environment import MazeEnvironment


def make_open_3x3():
    m = MazeEnvironment(3, 3, (0, 0), (2, 2))
    for x in range(3):
        for y in range(3):
            for nx, ny in [(x + 1, y), (x, y + 1)]:
                if 0 <= nx < 3 and 0 <= ny < 3:
                    m.remove_wall((x, y), (nx, ny))
    return m


def make_ga():
    return GeneticAlgorithm(make_open_3x3(), pop_size=50)


def sample_chromosomes():
    random.seed(42)
    ga = make_ga()
    population = ga._initialize_population()

    crossover_samples = []
    mutate_samples = []
    for _ in range(50):
        parent_a = random.choice(population)
        parent_b = random.choice(population)
        crossover_samples.append(ga._crossover(parent_a, parent_b)["path"])

        individual = random.choice(population)
        mutate_samples.append(ga._mutate(individual)["path"])

    paths = [ind["path"] for ind in population] + crossover_samples + mutate_samples
    return ga, paths


def test_path_starts_at_maze_start():
    ga, paths = sample_chromosomes()
    assert len(paths) >= 150
    for path in paths:
        assert path[0] == ga.maze.start


def test_consecutive_cells_are_adjacent():
    ga, paths = sample_chromosomes()
    for path in paths:
        for left, right in zip(path, path[1:]):
            assert right in ga.maze.neighbors(*left)


def test_no_walls_between_consecutive_cells():
    ga, paths = sample_chromosomes()
    for path in paths:
        for left, right in zip(path, path[1:]):
            assert not ga.maze.has_wall_between(left, right)


def test_no_duplicate_cells_in_path():
    ga, paths = sample_chromosomes()
    for path in paths:
        assert len(set(path)) == len(path)


def test_path_length_within_bounds():
    ga, paths = sample_chromosomes()
    for path in paths:
        assert 1 <= len(path) <= ga.max_path_length
