import random
from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment

class BaselineGA:
    """
    (1+1)-GA baseline: A randomized approach that can be configured 
    as a systematic DFS (backtrack=True) or a naive random walk (backtrack=False).
    """

    def __init__(self, maze: MazeEnvironment, backtrack: bool = True):
        self.maze = maze
        self.backtrack = backtrack
        self.entropy_history: list[float] = []
        # Track all cells explored across the entire run
        self.visited = set([self.maze.start])

    def _initialize_individual(self) -> list[tuple]:
        return [self.maze.start]

    def _mutate(self, path: list[tuple]) -> list[tuple]:
        """
        Moves to an unvisited neighbor. If all neighbors are visited,
        it either backtracks or stays stuck based on initialization.
        """
        if not path:
            return [self.maze.start]

        x, y = path[-1]
        neighbors = []

        # Check all four directions for valid moves (no walls)
        if x + 1 < self.maze.width and not self.maze.vertical_walls[y, x + 1]:
            neighbors.append((x + 1, y))
        if x - 1 >= 0 and not self.maze.vertical_walls[y, x]:
            neighbors.append((x - 1, y))
        if y + 1 < self.maze.height and not self.maze.horizontal_walls[y + 1, x]:
            neighbors.append((x, y + 1))
        if y - 1 >= 0 and not self.maze.horizontal_walls[y, x]:
            neighbors.append((x, y - 1))

        # Filter for neighbors we haven't visited yet
        options = [n for n in neighbors if n not in self.visited]

        if options:
            next_step = random.choice(options)
            self.visited.add(next_step)
            path.append(next_step)
        elif self.backtrack:
            # Backtrack if enabled and we are surrounded by visited cells or walls
            if len(path) > 1:
                path.pop()
        
        # If backtrack is False and there are no options, the agent stays 
        # in the current cell until the end of the simulation.
        return path

    def run(self, max_iterations: int = 1000) -> dict:
        individual = self._initialize_individual()

        for iteration in range(max_iterations):
            individual = self._mutate(individual)

            if iteration % 10 == 0:
                self.entropy_history.append(calculate_shannon_entropy([individual[-1]]))

            if individual[-1] == self.maze.goal:
                return {"success": True, "iterations": iteration + 1, "path": individual}

        return {"success": False, "iterations": max_iterations, "path": individual}