import random
from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment

# Used to break Matplotlib lines so they don't zig zag across the maze
NAN_NODE = (float('nan'), float('nan'))

class GeneticAlgorithm:
    """
    Primary GA for maze navigation.
    
    Features:
    - Tournament selection (K=5)
    - Path segment crossover (splicing at common nodes)
    - 4 neighbor local search mutation (adapted from 8 neighbor to respect grid walls)
    """

    def __init__(self, maze: MazeEnvironment, pop_size: int = 50, tournament_k: int = 5):
        self.maze = maze
        self.pop_size = pop_size
        self.tournament_k = tournament_k
        self.entropy_history: list[float] = []
        self.global_history: list[tuple] = [] 

    def initialize_population(self) -> list[dict]:
        """Initialize the population with start nodes."""
        return [
            {
                "path": [self.maze.start], 
                "visited": {self.maze.start}, 
                "history": [self.maze.start]
            }
            for _ in range(self.pop_size)
        ]

    def fitness(self, path: list[tuple]) -> float:
        """
        Score a path using negative Manhattan distance. 
        Closer to the goal = less negative (higher fitness).
        """
        if not path:
            return -float('inf')
        x, y = path[-1]
        gx, gy = self.maze.goal
        dist = abs(x - gx) + abs(y - gy)
        return -dist

    def tournament_select(self, population: list[dict]) -> dict:
        """Sample tournament_k individuals, return the fittest."""
        participants = random.sample(population, self.tournament_k)
        return max(participants, key=lambda ind: self.fitness(ind["path"]))

    def crossover(self, parent_a: dict, parent_b: dict) -> dict:
        """
        Find a shared cell and splice path segments at that point.
        Includes a loop erasure mechanism to ensure the resulting child path is valid.
        """
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

    def mutate(self, child: dict) -> dict:
        """
        Extend path by one valid unvisited step. Backtrack if completely stuck.
        Records every step to the global history for the heatmap.
        """
        if not child["path"]:
            return child
            
        curr_x, curr_y = child["path"][-1]
        neighbors = self.maze.neighbors(curr_x, curr_y)
        options = [n for n in neighbors if n not in child["visited"]]

        if options:
            next_step = random.choice(options)
            child["visited"].add(next_step)
            child["path"].append(next_step)
            child["history"].append(next_step)
            
            # Record forward movement
            self.global_history.append((curr_x, curr_y))
            self.global_history.append(next_step)
            self.global_history.append(NAN_NODE)
        else:
            if len(child["path"]) > 1:
                prev_step = child["path"][-1]
                child["path"].pop()
                child["history"].append(child["path"][-1])
                
                # Record backward movement
                self.global_history.append(prev_step)
                self.global_history.append(child["path"][-1])
                self.global_history.append(NAN_NODE)
                
        return child

    def run(self, max_iterations: int = 1000, disruption_length: int = -1) -> dict:
        population = self.initialize_population()
        self.global_history = [] 
        
        snapshot_path = None
        wall_dropped = False
        disruption_iteration = None

        for iteration in range(max_iterations):
            
            # --- SPATIAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_length > 0:
                if any(len(ind["path"]) >= disruption_length for ind in population):
                    if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                        wall_dropped = True
                        
                        fittest = max(population, key=lambda ind: self.fitness(ind["path"]))
                        snapshot_path = fittest["path"][:]
                        disruption_iteration = iteration
                        
                        w1, w2 = self.maze.dynamic_wall
                        self.maze.add_wall(w1, w2)
                        
                        for ind in population:
                            path = ind["path"]
                            for i in range(len(path) - 1):
                                if (path[i] == w1 and path[i+1] == w2) or \
                                   (path[i] == w2 and path[i+1] == w1):
                                    ind["path"] = path[:i+1]
                                    ind["visited"] = set(ind["path"])
                                    ind["history"].append(ind["path"][-1])
                                    break
            
            if iteration % 10 == 0:
                positions = [ind["path"][-1] for ind in population]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            new_population = []
            for _ in range(self.pop_size):
                parent_a = self.tournament_select(population)
                parent_b = self.tournament_select(population)
                
                child = self.crossover(parent_a, parent_b)
                child = self.mutate(child)
                
                new_population.append(child)
                
            population = new_population

            for individual in population:
                if individual["path"][-1] == self.maze.goal:
                    return {
                        "success": True, 
                        "iterations": iteration + 1, 
                        "path": individual["path"],
                        "snapshot": snapshot_path,
                        "disruption_iteration": disruption_iteration,
                        "history": self.global_history 
                    }

        fittest = max(population, key=lambda ind: self.fitness(ind["path"]))
        return {
            "success": False, 
            "iterations": max_iterations, 
            "path": fittest["path"],
            "snapshot": snapshot_path,
            "disruption_iteration": disruption_iteration,
            "history": self.global_history
        }