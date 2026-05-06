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

    def run(self, max_iterations: int = 1000, disruption_length: int = -1) -> dict:
        individual = self._initialize_individual()
        snapshot_path = None
        wall_dropped = False
        disruption_iteration = None # NEW: Track the exact iteration

        for iteration in range(max_iterations):
            
            # --- SPATIAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_length > 0 and len(individual) >= disruption_length:
                if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                    wall_dropped = True
                    snapshot_path = list(individual)
                    disruption_iteration = iteration # NEW: Save the iteration
                    
                    w1, w2 = self.maze.dynamic_wall
                    self.maze.add_wall(w1, w2)
                    
                    for i in range(len(individual) - 1):
                        if (individual[i] == w1 and individual[i+1] == w2) or \
                           (individual[i] == w2 and individual[i+1] == w1):
                            individual = individual[:i+1]
                            self.visited = set(individual) 
                            break

            individual = self._mutate(individual)

            if iteration % 10 == 0:
                self.entropy_history.append(calculate_shannon_entropy([individual[-1]]))

            if individual[-1] == self.maze.goal:
                return {
                    "success": True, "iterations": iteration + 1, 
                    "path": individual, "snapshot": snapshot_path, 
                    "disruption_iteration": disruption_iteration
                }

        return {
            "success": False, "iterations": max_iterations, 
            "path": individual, "snapshot": snapshot_path, 
            "disruption_iteration": disruption_iteration
        }