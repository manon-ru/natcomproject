from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment


class GeneticAlgorithm:
    """
    Primary GA: tournament selection (K=5), path-segment crossover,
    and 8-neighbor local search mutation.
    """

    def __init__(self, maze: MazeEnvironment, pop_size: int = 50, tournament_k: int = 5):
        self.maze = maze
        self.pop_size = pop_size
        self.tournament_k = tournament_k
        self.entropy_history: list[float] = []

    def _initialize_population(self) -> list[list[tuple]]:
        return [[self.maze.start] for _ in range(self.pop_size)]

    def _fitness(self, path: list[tuple]) -> float:
        # TODO: score a path; suggested: negative Manhattan distance to goal
        raise NotImplementedError

    def _tournament_select(self, population: list[list[tuple]]) -> list[tuple]:
        # TODO: sample tournament_k individuals, return the fittest
        raise NotImplementedError

    def _crossover(self, parent_a: list[tuple], parent_b: list[tuple]) -> list[tuple]:
        # TODO: find a shared cell and splice path segments at that point
        raise NotImplementedError

    def _mutate(self, path: list[tuple]) -> list[tuple]:
        # TODO: extend path by one valid step using 8-neighbor local search; backtrack if stuck
        raise NotImplementedError

    def run(self, max_iterations: int = 1000) -> dict:
        population = self._initialize_population()

        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                positions = [ind[-1] for ind in population]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            new_population = []
            for _ in range(self.pop_size):
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                new_population.append(child)
            population = new_population

            for individual in population:
                if individual[-1] == self.maze.goal:
                    return {"success": True, "iterations": iteration + 1, "path": individual}

        return {"success": False, "iterations": max_iterations, "path": None}
