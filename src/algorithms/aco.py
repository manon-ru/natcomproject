"""
Ant Colony Optimization for grid-based maze navigation.

NOTE ON PHEROMONE REPRESENTATION: This implementation stores pheromone
values per CELL rather than per EDGE. Akka & Khaber (cited in the proposal
for structural parameters) typically use a per-edge formulation. Our
per-cell choice is a project-specific design that reduces memory and is
consistent with the cell-by-cell agent movement model. This deviation is
noted in the report's Approach section.
"""
import random
import numpy as np
from evaluation.metrics import calculate_shannon_entropy
from maze.environment import MazeEnvironment

# Marker for Matplotlib to break line segments
NAN_NODE = (float('nan'), float('nan'))

class ACO:
    def __init__(
        self,
        maze: MazeEnvironment,
        num_ants: int = 50,
        evaporation_rate: float = 0.1,
        pheromone_deposit: float = 2.0,
        alpha: float = 1.0,
        beta: float = 5.0,
        initial_pheromone: float = 0.8,
    ):
        self.maze = maze
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.alpha = alpha
        self.beta = beta
        self.initial_pheromone = initial_pheromone
        self.entropy_history: list[float] = []
        self.global_history: list[tuple] = []   # Tracks all movement
        self.snapshot_history: list[tuple] = [] # Collective state at T=100
        
        self.pheromones = np.ones((maze.height, maze.width), dtype=float) * initial_pheromone

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

    def run(self, max_iterations: int = 1000, disruption_iteration: int = -1, forced_min_iterations: int = 0) -> dict:
        ants = self.initialize_ants()
        self.global_history = []
        
        snapshot_path = None
        wall_dropped = False
        disruption_iteration_recorded = None
        
        # NEW: Store the first successful run so we can return it later
        first_success_result = None 

        for iteration in range(max_iterations):
            
            # --- TEMPORAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_iteration > 0 and iteration == disruption_iteration:
                if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                    wall_dropped = True
                    
                    self.snapshot_history = self.global_history[:]
                    
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

            # Record entropy every 10 iterations
            if iteration % 10 == 0:
                positions = [a["path"][-1] for a in ants]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            self.evaporate()

            for ant in ants:
                curr_pos = ant["path"][-1]
                
                # Freeze ant if it reached the goal so it doesn't wander off
                if curr_pos == self.maze.goal and wall_dropped:
                    continue
                    
                next_cell = self.choose_next(curr_pos[0], curr_pos[1], ant["visited"])

                if next_cell:
                    ant["visited"].add(next_cell)
                    ant["path"].append(next_cell)
                    
                    self.global_history.append(curr_pos)
                    self.global_history.append(next_cell)
                    self.global_history.append(NAN_NODE)
                    
                    self.pheromones[next_cell[1], next_cell[0]] += self.pheromone_deposit
                else:
                    if len(ant["path"]) > 1:
                        old_pos = ant["path"].pop()
                        self.global_history.append(old_pos)
                        self.global_history.append(ant["path"][-1])
                        self.global_history.append(NAN_NODE)

                if ant["path"][-1] == self.maze.goal:
                    if disruption_iteration > 0 and iteration < disruption_iteration:
                        snapshot_path = ant["path"][:] 
                        ant["path"] = [self.maze.start]
                        ant["visited"] = {self.maze.start}
                    else:
                        # NEW: Record success but don't break immediately
                        if first_success_result is None:
                            first_success_result = {
                                "success": True, 
                                "iterations": iteration + 1, 
                                "path": ant["path"][:],
                                "snapshot": snapshot_path,
                                "snapshot_history": self.snapshot_history,
                                "disruption_iteration": disruption_iteration_recorded,
                            }

            # NEW: Only return if we have a success AND we've passed the forced minimum
            if first_success_result is not None and iteration >= forced_min_iterations:
                first_success_result["history"] = self.global_history
                return first_success_result

        # Fallback if loop ends
        if first_success_result is not None:
            first_success_result["history"] = self.global_history
            return first_success_result

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
