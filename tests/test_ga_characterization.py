import random

from algorithms.ga import GeneticAlgorithm
from tests.fixtures.small_mazes import trivial_3x3


def make_open_3x3():
    return trivial_3x3()


def test_initial_population_size_matches_pop_size():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze, pop_size=7)

    population = ga._initialize_population()

    assert len(population) == 7


def test_initial_paths_start_at_maze_start():
    random.seed(0)
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze, pop_size=7)

    population = ga._initialize_population()

    assert all(ind["path"][0] == maze.start for ind in population)


def test_random_walk_does_not_revisit():
    random.seed(0)
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)

    for _ in range(20):
        path = ga._random_walk([maze.start])
        assert len(path) == len(set(path))


def test_truncate_at_goal_returns_prefix_through_goal():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)]

    truncated = ga._truncate_at_goal(path)

    assert truncated == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert truncated[-1] == maze.goal


def test_truncate_at_goal_returns_full_path_when_no_goal():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)
    path = [(0, 0), (1, 0), (1, 1), (1, 2)]

    assert ga._truncate_at_goal(path) == path


def test_reached_goal_true_when_goal_in_path():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)

    assert ga._reached_goal({"path": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]}) is True


def test_reached_goal_false_when_goal_not_in_path():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)

    assert ga._reached_goal({"path": [(0, 0), (1, 0), (1, 1)]}) is False


def test_fitness_goal_reaching_higher_than_non_reaching():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)
    goal_path = {"path": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]}
    non_goal_path = {"path": [(0, 0), (1, 0), (1, 1)]}

    assert ga._fitness(goal_path) > ga._fitness(non_goal_path)


def test_fitness_shorter_goal_path_scores_higher():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)
    shorter = {"path": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]}
    longer = {"path": [(0, 0), (1, 0), (2, 0), (1, 0), (2, 0), (2, 1), (2, 2)]}

    assert ga._fitness(shorter) > ga._fitness(longer)


def test_fitness_closer_to_goal_scores_higher_among_non_reachers():
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)
    closer = {"path": [(0, 0), (1, 0), (2, 0), (2, 1)]}
    farther = {"path": [(0, 0), (0, 1)]}

    assert ga._fitness(closer) > ga._fitness(farther)


def test_roulette_select_returns_individual_from_population():
    random.seed(0)
    maze = make_open_3x3()
    ga = GeneticAlgorithm(maze)
    population = [
        {"path": [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]},
        {"path": [(0, 0), (0, 1), (0, 2)]},
        {"path": [(0, 0), (1, 0), (1, 1)]},
    ]

    selected = ga._roulette_select(population)

    assert selected in population
