from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment


class BaselineGA:
    """
    (1+1)-GA baseline: a single individual that mutates one step at a time
    and backtracks when stuck. Serves as the control to verify that complex
    population dynamics actually help over simple mutation + backtracking.
    """

    def __init__(self, maze: MazeEnvironment):
        self.maze = maze
        self.entropy_history: list[float] = []

    def _initialize_individual(self) -> list[tuple]:
        return [self.maze.start]

    def _mutate(self, path: list[tuple]) -> list[tuple]:
        # TODO: extend path by one valid step; backtrack if stuck
        raise NotImplementedError

    def run(self, max_iterations: int = 1000) -> dict:
        individual = self._initialize_individual()

        for iteration in range(max_iterations):
            individual = self._mutate(individual)

            if iteration % 10 == 0:
                self.entropy_history.append(calculate_shannon_entropy([individual[-1]]))

            if individual[-1] == self.maze.goal:
                return {"success": True, "iterations": iteration + 1, "path": individual}

        return {"success": False, "iterations": max_iterations, "path": None}
