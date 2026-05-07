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
        self.visited = set([self.maze.start])
        self.full_history = [self.maze.start]

    def initialize_individual(self) -> list[tuple]:
        return [self.maze.start]

    def mutate(self, path: list[tuple], disruption_iteration: int = -1) -> list[tuple]:
        if not path:
            return [self.maze.start]

        # GATE: Only freeze if we are in Phase 1 of a Sudden Wall experiment.
        # This allows the baseline to finish early in non-dynamic tiers.
        if disruption_iteration > 0 and path[-1] == self.maze.goal:
            return path

        x, y = path[-1]
        neighbors = []

        # Logic for determining valid neighbors based on maze walls.
        if x + 1 < self.maze.width and not self.maze.vertical_walls[y, x + 1]:
            neighbors.append((x + 1, y))
        if x - 1 >= 0 and not self.maze.vertical_walls[y, x]:
            neighbors.append((x - 1, y))
        if y + 1 < self.maze.height and not self.maze.horizontal_walls[y + 1, x]:
            neighbors.append((x, y + 1))
        if y - 1 >= 0 and not self.maze.horizontal_walls[y, x]:
            neighbors.append((x, y - 1))

        options = [n for n in neighbors if n not in self.visited]

        if options:
            next_step = random.choice(options)
            self.visited.add(next_step)
            path.append(next_step)
            self.full_history.append(next_step) 
        elif self.backtrack:
            if len(path) > 1:
                path.pop()
                self.full_history.append(path[-1]) 
        else:
            self.full_history.append(path[-1]) 

        return path

    def run(self, max_iterations: int = 1000, disruption_iteration: int = -1) -> dict:
        individual = self.initialize_individual()
        snapshot_path = None
        wall_dropped = False
        disruption_iteration_recorded = None

        for iteration in range(max_iterations):
            # --- TEMPORAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_iteration > 0 and iteration == disruption_iteration:
                if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                    wall_dropped = True
                    snapshot_path = list(individual)
                    disruption_iteration_recorded = iteration
                    
                    w1, w2 = self.maze.dynamic_wall
                    self.maze.add_wall(w1, w2)
                    
                    # Cut the path if the new wall bisects the agent's current route.
                    for i in range(len(individual) - 1):
                        if (individual[i] == w1 and individual[i+1] == w2) or \
                           (individual[i] == w2 and individual[i+1] == w1):
                            individual = individual[:i+1]
                            self.visited = set(individual) 
                            self.full_history.append(individual[-1])
                            break

            # Pass disruption_iteration to handle the conditional goal freeze.
            individual = self.mutate(individual, disruption_iteration)

            if iteration % 10 == 0:
                self.entropy_history.append(calculate_shannon_entropy([individual[-1]]))

            if individual[-1] == self.maze.goal:
                if disruption_iteration > 0 and iteration < disruption_iteration:
                    continue # PHASE 1: Stay at goal to reinforce location.
                else:
                    # PHASE 2: Goal found or disruption time passed.
                    return {
                        "success": True, 
                        "iterations": iteration + 1, 
                        "path": individual, 
                        "snapshot": snapshot_path, 
                        "disruption_iteration": disruption_iteration_recorded,
                        "history": self.full_history
                    }

        return {
            "success": False, 
            "iterations": max_iterations, 
            "path": individual, 
            "snapshot": snapshot_path, 
            "disruption_iteration": disruption_iteration_recorded,
            "history": self.full_history
        }