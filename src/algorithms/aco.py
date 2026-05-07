import random
import numpy as np
from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment

# Marker for Matplotlib to break line segments
NAN_NODE = (float('nan'), float('nan'))

class ACO:
    def __init__(
        self,
        maze: MazeEnvironment,
        num_ants: int = 50,
        evaporation_rate: float = 0.1,
        pheromone_deposit: float = 1.0,
        alpha: float = 1.0,
        beta: float = 2.0,
    ):
        self.maze = maze
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.alpha = alpha
        self.beta = beta
        self.entropy_history: list[float] = []
        self.global_history: list[tuple] = []   # Tracks all movement
        self.snapshot_history: list[tuple] = [] # Collective state at T=100
        
        self.pheromones = np.ones((maze.height, maze.width), dtype=float) * 0.1

    def initialize_ants(self) -> list[dict]:
        return [
            {
                "path": [self.maze.start], 
                "visited": {self.maze.start}, 
                "history": [self.maze.start]
            }
            for _ in range(self.num_ants)
        ]

    def heuristic(self, x: int, y: int) -> float:
        gx, gy = self.maze.goal
        dist = abs(x - gx) + abs(y - gy)
        return 1.0 / (dist + 1.0)

    def choose_next(self, x: int, y: int, visited: set) -> tuple | None:
        neighbors = self.maze.neighbors(x, y)
        options = [n for n in neighbors if n not in visited]

        if not options:
            return None

        weights = []
        for nx, ny in options:
            tau = self.pheromones[ny, nx] ** self.alpha
            eta = self.heuristic(nx, ny) ** self.beta
            weights.append(tau * eta)

        total_weight = sum(weights)
        if total_weight == 0:
            probs = [1.0 / len(options)] * len(options)
        else:
            probs = [w / total_weight for w in weights]

        idx = np.random.choice(len(options), p=probs)
        return options[idx]

    def evaporate(self) -> None:
        self.pheromones *= (1.0 - self.evaporation_rate)
        self.pheromones = np.maximum(self.pheromones, 0.01)

    def run(self, max_iterations: int = 1000, disruption_iteration: int = -1) -> dict:
        ants = self.initialize_ants()
        self.global_history = []
        
        snapshot_path = None
        wall_dropped = False
        disruption_iteration_recorded = None

        for iteration in range(max_iterations):
            
            # --- TEMPORAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_iteration > 0 and iteration == disruption_iteration:
                if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                    wall_dropped = True
                    
                    # Capture the collective swarm cloud
                    self.snapshot_history = self.global_history[:]
                    
                    # Find the most complete path currently in the swarm for the red line
                    best_ant = min(ants, key=lambda a: self.heuristic(*a["path"][-1]))
                    snapshot_path = best_ant["path"][:]
                    disruption_iteration_recorded = iteration
                    
                    w1, w2 = self.maze.dynamic_wall
                    self.maze.add_wall(w1, w2)
                    
                    for a in ants:
                        path = a["path"]
                        for i in range(len(path) - 1):
                            if (path[i] == w1 and path[i+1] == w2) or \
                               (path[i] == w2 and path[i+1] == w1):
                                a["path"] = path[:i+1]
                                a["visited"] = set(a["path"])
                                break

            if iteration % 10 == 0:
                positions = [a["path"][-1] for a in ants]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            self.evaporate()

            for ant in ants:
                curr_pos = ant["path"][-1]
                next_cell = self.choose_next(curr_pos[0], curr_pos[1], ant["visited"])

                if next_cell:
                    ant["visited"].add(next_cell)
                    ant["path"].append(next_cell)
                    
                    # Record for global heatmap
                    self.global_history.append(curr_pos)
                    self.global_history.append(next_cell)
                    self.global_history.append(NAN_NODE)
                    
                    self.pheromones[next_cell[1], next_cell[0]] += self.pheromone_deposit
                else:
                    if len(ant["path"]) > 1:
                        old_pos = ant["path"].pop()
                        # Record backtrack
                        self.global_history.append(old_pos)
                        self.global_history.append(ant["path"][-1])
                        self.global_history.append(NAN_NODE)

                if ant["path"][-1] == self.maze.goal:
                    if disruption_iteration > 0 and iteration < disruption_iteration:
                        # Before wall: Save the successful path and reset
                        snapshot_path = ant["path"][:] 
                        ant["path"] = [self.maze.start]
                        ant["visited"] = {self.maze.start}
                    else:
                        # After wall: Final success
                        return {
                            "success": True, 
                            "iterations": iteration + 1, 
                            "path": ant["path"],
                            "snapshot": snapshot_path,
                            "snapshot_history": self.snapshot_history,
                            "disruption_iteration": disruption_iteration_recorded,
                            "history": self.global_history
                        }

        best_ant = min(ants, key=lambda a: self.heuristic(*a["path"][-1]))
        return {
            "success": False, 
            "iterations": max_iterations, 
            "path": best_ant["path"],
            "snapshot": snapshot_path,
            "snapshot_history": self.snapshot_history,
            "disruption_iteration": disruption_iteration_recorded,
            "history": self.global_history
        }