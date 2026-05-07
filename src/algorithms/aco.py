import random
import numpy as np

from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment


class ACO:
    """
    Ant Colony Optimization for maze navigation (Step-by-Step Online Update).

    Ants navigate cell-by-cell. At each step, they deposit pheromone directly 
    onto their current cell. Future ants bias movement toward high-pheromone 
    cells combined with a heuristic (distance to goal).
    """

    def __init__(
        self,
        maze: MazeEnvironment,
        num_ants: int = 50,
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
        
        # Initialize pheromones to a baseline value to encourage initial exploration
        self.pheromones = np.ones((maze.height, maze.width), dtype=float) * 0.1

    def _initialize_ants(self) -> list[dict]:
        return [
            {
                "path": [self.maze.start], 
                "visited": {self.maze.start}, 
                "history": [self.maze.start]
            }
            for _ in range(self.num_ants)
        ]

    def heuristic(self, x: int, y: int) -> float:
        """Desirability score based on inverse Manhattan distance to goal."""
        gx, gy = self.maze.goal
        dist = abs(x - gx) + abs(y - gy)
        return 1.0 / (dist + 1.0)

    def choose_next(self, x: int, y: int, visited: set) -> tuple | None:
        """Probabilistically select the next cell using pheromone and heuristic."""
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
        
        # Fallback to uniform random if all weights collapse to 0
        if total_weight == 0:
            probs = [1.0 / len(options)] * len(options)
        else:
            probs = [w / total_weight for w in weights]

        # Roulette wheel selection using numpy
        idx = np.random.choice(len(options), p=probs)
        return options[idx]

    def evaporate(self) -> None:
        """Decay all pheromones and enforce a minimum floor to prevent 0 probabilities."""
        self.pheromones *= (1.0 - self.evaporation_rate)
        self.pheromones = np.maximum(self.pheromones, 0.01)

    def run(self, max_iterations: int = 1000, disruption_length: int = -1) -> dict:
        ants = self._initialize_ants()
        
        snapshot_path = None
        wall_dropped = False
        disruption_iteration = None

        for iteration in range(max_iterations):
            
            # --- SPATIAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_length > 0:
                for ant in ants:
                    if len(ant["path"]) >= disruption_length:
                        if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                            wall_dropped = True
                            snapshot_path = ant["path"][:]
                            disruption_iteration = iteration
                            
                            w1, w2 = self.maze.dynamic_wall
                            self.maze.add_wall(w1, w2)
                            
                            # Sever paths for any ant crossing the new wall
                            for a in ants:
                                path = a["path"]
                                for i in range(len(path) - 1):
                                    if (path[i] == w1 and path[i+1] == w2) or \
                                       (path[i] == w2 and path[i+1] == w1):
                                        a["path"] = path[:i+1]
                                        a["visited"] = set(a["path"])
                                        a["history"].append(a["path"][-1])
                                        break
                            break

            # 1. Track Spatial Diversity
            if iteration % 10 == 0:
                positions = [a["path"][-1] for a in ants]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            # 2. Environmental Pheromone Evaporation
            self.evaporate()

            # 3. Move all ants one step
            for ant in ants:
                curr_x, curr_y = ant["path"][-1]
                next_cell = self.choose_next(curr_x, curr_y, ant["visited"])

                if next_cell:
                    ant["visited"].add(next_cell)
                    ant["path"].append(next_cell)
                    ant["history"].append(next_cell)
                    
                    # Online Deposit: Leave a trail immediately
                    self.pheromones[next_cell[1], next_cell[0]] += self.pheromone_deposit
                else:
                    # Backtrack if stuck in a dead end
                    if len(ant["path"]) > 1:
                        ant["path"].pop()
                        ant["history"].append(ant["path"][-1])

                # 4. Check for success
                if ant["path"][-1] == self.maze.goal:
                    return {
                        "success": True, 
                        "iterations": iteration + 1, 
                        "path": ant["path"],
                        "snapshot": snapshot_path,
                        "disruption_iteration": disruption_iteration,
                        "history": ant["history"]
                    }

        # Return best effort if time runs out (Ant closest to goal)
        best_ant = min(ants, key=lambda a: self.heuristic(*a["path"][-1]))
        return {
            "success": False, 
            "iterations": max_iterations, 
            "path": best_ant["path"],
            "snapshot": snapshot_path,
            "disruption_iteration": disruption_iteration,
            "history": best_ant["history"]
        }