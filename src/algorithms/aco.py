import numpy as np

from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment


class ACO:
    """
    Ant Colony Optimization for maze navigation.

    Ants deposit pheromone directly onto grid cells (indexed [y, x]).
    Future ants bias movement toward high-pheromone cells.
    Pheromone grid initialized to 0.1.
    """

    def __init__(
        self,
        maze: MazeEnvironment,
        num_ants: int = 30,
        evaporation_rate: float = 0.1,
        pheromone_deposit: float = 1.0,
        alpha: float = 1.0,  # Pheromone influence weight
        beta: float = 2.0,   # Heuristic influence weight
    ):
        self.maze = maze
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.alpha = alpha
        self.beta = beta
        self.entropy_history: list[float] = []
        self.pheromones = np.ones((maze.height, maze.width), dtype=float) * 0.1

    def _heuristic(self, x: int, y: int) -> float:
        # TODO: return a desirability score for cell (x, y); suggested: 1 / (distance to goal + 1)
        raise NotImplementedError

    def _choose_next(self, x: int, y: int, visited: set) -> tuple | None:
        # TODO: pick next cell probabilistically using pheromone^alpha * heuristic^beta weights
        raise NotImplementedError

    def _evaporate(self) -> None:
        # TODO: reduce all pheromone values by evaporation_rate; enforce a minimum floor
        raise NotImplementedError

    def _deposit(self, path: list[tuple]) -> None:
        # TODO: add pheromone to each cell in path proportional to pheromone_deposit / path length
        raise NotImplementedError

    def _run_ant(self) -> list[tuple]:
        # TODO: walk one ant from start to goal (or until stuck), return the path taken
        raise NotImplementedError

    def run(self, max_iterations: int = 1000) -> dict:
        for iteration in range(max_iterations):
            ant_paths = [self._run_ant() for _ in range(self.num_ants)]

            if iteration % 10 == 0:
                positions = [path[-1] for path in ant_paths]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            self._evaporate()

            for path in ant_paths:
                if path[-1] == self.maze.goal:
                    self._deposit(path)
                    return {"success": True, "iterations": iteration + 1, "path": path}

        return {"success": False, "iterations": max_iterations, "path": None}
