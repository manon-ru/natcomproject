import random
from evaluation.metrics import calculate_shannon_entropy
from maze.environment import MazeEnvironment

# Used to break Matplotlib lines so they don't zig zag across the maze
NAN_NODE = (float('nan'), float('nan'))

class GeneticAlgorithm:
    """
    Primary GA for maze navigation.
    """

    def __init__(self, maze: MazeEnvironment, pop_size: int = 50, tournament_k: int = 5):
        self.maze = maze
        self.pop_size = pop_size
        self.tournament_k = tournament_k
        self.entropy_history: list[float] = []
        self.global_history: list[tuple] = [] 
        self.snapshot_history: list[tuple] = [] # Captured at T=100

    def initialize_population(self) -> list[dict]:
        return [
            {
                "path": [self.maze.start], 
                "visited": {self.maze.start}, 
                "history": [self.maze.start]
            }
            for _ in range(self.pop_size)
        ]

    def fitness(self, path: list[tuple]) -> float:
        if not path:
            return -float('inf')
        x, y = path[-1]
        gx, gy = self.maze.goal
        dist = abs(x - gx) + abs(y - gy)
        return -dist

    def tournament_select(self, population: list[dict]) -> dict:
        participants = random.sample(population, self.tournament_k)
        return max(participants, key=lambda ind: self.fitness(ind["path"]))

    def crossover(self, parent_a: dict, parent_b: dict) -> dict:
        path_a = parent_a["path"]
        path_b = parent_b["path"]
        
        common_nodes = list(set(path_a) & set(path_b))
        splice_point = random.choice(common_nodes)
        
        idx_a = path_a.index(splice_point)
        idx_b = path_b.index(splice_point)
        
        raw_path = path_a[:idx_a] + path_b[idx_b:]
        
        clean_path = []
        for node in raw_path:
            if node in clean_path:
                idx = clean_path.index(node)
                clean_path = clean_path[:idx]
            clean_path.append(node)
            
        new_visited = parent_a["visited"].copy()
        new_visited.update(clean_path)
            
        return {
            "path": clean_path,
            "visited": new_visited,
            "history": parent_a["history"][:idx_a] + clean_path[idx_a:] 
        }

    def mutate(self, child: dict, disruption_iteration: int = -1) -> dict:
        """
        Gated mutation: only freezes at the goal if a disruption is scheduled.
        """
        if not child["path"]:
            return child
        
        # GATE: Only freeze if we are in Phase 1 of a Sudden Wall experiment
        if disruption_iteration > 0 and child["path"][-1] == self.maze.goal:
            return child
            
        curr_x, curr_y = child["path"][-1]
        neighbors = self.maze.neighbors(curr_x, curr_y)
        options = [n for n in neighbors if n not in child["visited"]]

        if options:
            next_step = random.choice(options)
            child["visited"].add(next_step)
            child["path"].append(next_step)
            
            # Record move to global history for the heatmap lines
            self.global_history.append((curr_x, curr_y))
            self.global_history.append(next_step)
            self.global_history.append(NAN_NODE)
        else:
            if len(child["path"]) > 1:
                prev_step = child["path"][-1]
                child["path"].pop()
                
                # Record backtrack to global history
                self.global_history.append(prev_step)
                self.global_history.append(child["path"][-1])
                self.global_history.append(NAN_NODE)
                
        return child

    def run(self, max_iterations: int = 1000, disruption_iteration: int = -1, forced_min_iterations: int = 0) -> dict:
        population = self.initialize_population()
        self.global_history = [] 
        self.snapshot_history = []
        
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
                    
                    fittest = max(population, key=lambda ind: self.fitness(ind["path"]))
                    snapshot_path = fittest["path"][:]
                    self.snapshot_history = self.global_history[:]
                    disruption_iteration_recorded = iteration
                    
                    w1, w2 = self.maze.dynamic_wall
                    self.maze.add_wall(w1, w2)
                    
                    for ind in population:
                        path = ind["path"]
                        for i in range(len(path) - 1):
                            if (path[i] == w1 and path[i+1] == w2) or \
                               (path[i] == w2 and path[i+1] == w1):
                                ind["path"] = path[:i+1]
                                ind["visited"] = set(ind["path"])
                                break
            
            if iteration % 10 == 0:
                positions = [ind["path"][-1] for ind in population]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            new_population = []
            for _ in range(self.pop_size):
                parent_a = self.tournament_select(population)
                parent_b = self.tournament_select(population)
                
                child = self.crossover(parent_a, parent_b)
                child = self.mutate(child, disruption_iteration)
                
                new_population.append(child)
                
            population = new_population

            for individual in population:
                if individual["path"][-1] == self.maze.goal:
                    if disruption_iteration > 0 and iteration < disruption_iteration:
                        continue 
                    else:
                        # NEW: Record success but don't break immediately
                        if first_success_result is None:
                            first_success_result = {
                                "success": True, 
                                "iterations": iteration + 1, 
                                "path": individual["path"][:],
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

        fittest = max(population, key=lambda ind: self.fitness(ind["path"]))
        return {
            "success": False, 
            "iterations": max_iterations, 
            "path": fittest["path"],
            "snapshot": snapshot_path,
            "snapshot_history": self.snapshot_history,
            "disruption_iteration": disruption_iteration_recorded,
            "history": self.global_history
        }