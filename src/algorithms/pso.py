import random
from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment

class PSO:
    """
    Particle Swarm Optimization for maze navigation.

    Each particle maintains a current path and a personal best.
    The swarm shares a global best. Particles update their trajectories
    based on personal and swarm bests using a cost-minimization approach 
    adapted for discrete grid environments.
    """

    def __init__(self, maze: MazeEnvironment, num_particles: int = 50, c1: float = 0.1, c2: float = 0.2):
        self.maze = maze
        self.num_particles = num_particles
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.entropy_history: list[float] = []

    def initialize_particles(self) -> list[dict]:
        return [
            {
                "path": [self.maze.start], 
                "personal_best": [self.maze.start],
                "visited": {self.maze.start},
                "history": [self.maze.start]
            }
            for _ in range(self.num_particles)
        ]

    def distance_to_goal(self, path: list[tuple]) -> float:
        """Calculates the Manhattan distance from the path's head to the maze goal."""
        if not path:
            return float('inf')
        x, y = path[-1]
        gx, gy = self.maze.goal
        return abs(x - gx) + abs(y - gy)

    def update_particle(self, particle: dict, global_best: list[tuple]) -> dict:
        """
        Moves particle one step. The choice of neighbor is biased by distance 
        to personal_best and global_best.
        """
        if not particle["path"]:
            particle["path"] = [self.maze.start]
            return particle

        curr_pos = particle["path"][-1]
        
        # Determine valid neighbors (no walls)
        all_neighbors = self.maze.neighbors(*curr_pos)
        
        # Filter out already visited cells to prevent immediate trivial loops
        options = [n for n in all_neighbors if n not in particle["visited"]]

        if options:
            best_neighbor = None
            min_cost = float('inf')
            
            pb_pos = particle["personal_best"][-1]
            gb_pos = global_best[-1]

            # Evaluate each neighbor option
            for n in options:
                # 1. Distance to personal best
                dp = abs(n[0] - pb_pos[0]) + abs(n[1] - pb_pos[1])
                # 2. Distance to global best
                dg = abs(n[0] - gb_pos[0]) + abs(n[1] - gb_pos[1])
                
                r1, r2 = random.random(), random.random()
                
                # 3. Calculate cost. Lower cost means higher attraction.
                # The inertia (w=1.0) is represented by random noise to ensure exploration.
                inertia = random.random()
                cost = inertia + (self.c1 * r1 * dp) + (self.c2 * r2 * dg)

                if cost < min_cost:
                    min_cost = cost
                    best_neighbor = n

            next_step = best_neighbor
            particle["visited"].add(next_step)
            particle["path"].append(next_step)
            particle["history"].append(next_step)

            # Update personal best if the new position is closer to the true goal
            if self.distance_to_goal(particle["path"]) < self.distance_to_goal(particle["personal_best"]):
                particle["personal_best"] = particle["path"][:]
        else:
            # Backtrack if the particle is completely stuck in a dead end
            if len(particle["path"]) > 1:
                particle["path"].pop()
                particle["history"].append(particle["path"][-1])

        return particle

    def run(self, max_iterations: int = 1000, disruption_length: int = -1) -> dict:
        particles = self.initialize_particles()
        global_best: list[tuple] = [self.maze.start]
        
        snapshot_path = None
        wall_dropped = False
        disruption_iteration = None

        for iteration in range(max_iterations):
            
            # --- SPATIAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_length > 0:
                # Check if ANY particle has reached the disruption trigger length
                for p in particles:
                    if len(p["path"]) >= disruption_length:
                        if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                            wall_dropped = True
                            snapshot_path = p["path"][:]
                            disruption_iteration = iteration
                            
                            w1, w2 = self.maze.dynamic_wall
                            self.maze.add_wall(w1, w2)
                            
                            # Sever paths for ALL particles that crossed the new wall
                            for part in particles:
                                path = part["path"]
                                for i in range(len(path) - 1):
                                    if (path[i] == w1 and path[i+1] == w2) or \
                                       (path[i] == w2 and path[i+1] == w1):
                                        part["path"] = path[:i+1]
                                        part["visited"] = set(part["path"])
                                        part["history"].append(part["path"][-1])
                                        break
                            break 

            # Track spatial diversity
            if iteration % 10 == 0:
                positions = [p["path"][-1] for p in particles]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            # Update all particles
            for particle in particles:
                particle = self.update_particle(particle, global_best)

                # Update global best if this particle found a better position
                if self.distance_to_goal(particle["path"]) < self.distance_to_goal(global_best):
                    global_best = particle["path"][:]

                # Check for success
                if particle["path"][-1] == self.maze.goal:
                    return {
                        "success": True, 
                        "iterations": iteration + 1, 
                        "path": particle["path"], 
                        "snapshot": snapshot_path,
                        "disruption_iteration": disruption_iteration,
                        "history": particle["history"]
                    }

        # Return the best found path if max iterations are reached
        return {
            "success": False, 
            "iterations": max_iterations, 
            "path": global_best, 
            "snapshot": snapshot_path,
            "disruption_iteration": disruption_iteration,
            "history": particles[0]["history"]
        }