import random
from evaluation.metrics import calculate_shannon_entropy
from maze.environment import MazeEnvironment

# Used to break Matplotlib lines so they do not zig zag across the maze
NAN_NODE = (float('nan'), float('nan'))

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
        self.global_history: list[tuple] = []   # Collective history of the swarm
        self.snapshot_history: list[tuple] = [] # Swarm history at T=100

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
        if not path: return float('inf')
        x, y = path[-1]
        gx, gy = self.maze.goal
        return abs(x - gx) + abs(y - gy)

    def update_particle(self, particle: dict, global_best: list[tuple], disruption_iteration: int = -1) -> dict:
        """
        Moves particle one step. The choice of neighbor is biased by distance 
        to personal_best and global_best.
        """
        if not particle["path"]:
            particle["path"] = [self.maze.start]
            return particle
        
        # GATE: Only freeze at the goal if we are in Phase 1 of a disruption trial.
        # This prevents the swarm from wandering away while waiting for the wall to drop.
        if disruption_iteration > 0 and particle["path"][-1] == self.maze.goal:
            return particle

        curr_pos = particle["path"][-1]
        all_neighbors = self.maze.neighbors(*curr_pos)
        options = [n for n in all_neighbors if n not in particle["visited"]]

        if options:
            best_neighbor = None
            min_cost = float('inf')
            pb_pos = particle["personal_best"][-1]
            gb_pos = global_best[-1]

            for n in options:
                # Calculate Manhattan distances for the cost function
                dp = abs(n[0] - pb_pos[0]) + abs(n[1] - pb_pos[1])
                dg = abs(n[0] - gb_pos[0]) + abs(n[1] - gb_pos[1])
                
                r1, r2 = random.random(), random.random()
                inertia = random.random()
                
                # The trajectory is influenced by cognitive (c1) and social (c2) components.
                # Cost is defined as: $cost = \text{inertia} + (c_1 \cdot r_1 \cdot d_p) + (c_2 \cdot r_2 \cdot d_g)$
                cost = inertia + (self.c1 * r1 * dp) + (self.c2 * r2 * dg)

                if cost < min_cost:
                    min_cost = cost
                    best_neighbor = n

            next_step = best_neighbor
            particle["visited"].add(next_step)
            particle["path"].append(next_step)
            
            # Record move to global history for the heatmap lines
            self.global_history.append(curr_pos)
            self.global_history.append(next_step)
            self.global_history.append(NAN_NODE)

            if self.distance_to_goal(particle["path"]) < self.distance_to_goal(particle["personal_best"]):
                particle["personal_best"] = particle["path"][:]
        else:
            if len(particle["path"]) > 1:
                prev_pos = particle["path"][-1]
                particle["path"].pop()
                # Record backtrack to global history
                self.global_history.append(prev_pos)
                self.global_history.append(particle["path"][-1])
                self.global_history.append(NAN_NODE)

        return particle

    def run(self, max_iterations: int = 1000, disruption_iteration: int = -1, forced_min_iterations: int = 0) -> dict:
        particles = self.initialize_particles()
        global_best: list[tuple] = [self.maze.start]
        self.global_history = []
        
        snapshot_path = None
        wall_dropped = False
        disruption_iteration_recorded = None
        
        # Store the first successful run so we can return it later
        first_success_result = None

        for iteration in range(max_iterations):
            # --- TEMPORAL DISRUPTION LOGIC ---
            if not wall_dropped and disruption_iteration > 0 and iteration == disruption_iteration:
                if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                    wall_dropped = True
                    snapshot_path = global_best[:]
                    # Capture the collective history of all particles up to this point
                    self.snapshot_history = self.global_history[:]
                    disruption_iteration_recorded = iteration
                    
                    w1, w2 = self.maze.dynamic_wall
                    self.maze.add_wall(w1, w2)
                    
                    for part in particles:
                        path = part["path"]
                        for i in range(len(path) - 1):
                            if (path[i] == w1 and path[i+1] == w2) or \
                               (path[i] == w2 and path[i+1] == w1):
                                part["path"] = path[:i+1]
                                part["visited"] = set(part["path"])
                                break

            if iteration % 10 == 0:
                positions = [p["path"][-1] for p in particles]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            for particle in particles:
                particle = self.update_particle(particle, global_best, disruption_iteration)
                
                if self.distance_to_goal(particle["path"]) < self.distance_to_goal(global_best):
                    global_best = particle["path"][:]

                if particle["path"][-1] == self.maze.goal:
                    if disruption_iteration > 0 and iteration < disruption_iteration:
                        continue 
                    else:
                        if first_success_result is None:
                            first_success_result = {
                                "success": True, 
                                "iterations": iteration + 1, 
                                "path": particle["path"][:], 
                                "snapshot": snapshot_path,
                                "snapshot_history": self.snapshot_history,
                                "disruption_iteration": disruption_iteration_recorded,
                            }
                            
            # Only return if we have a success AND we have passed the forced minimum
            if first_success_result is not None and iteration >= forced_min_iterations:
                first_success_result["history"] = self.global_history
                return first_success_result

        # Fallback if loop ends
        if first_success_result is not None:
            first_success_result["history"] = self.global_history
            return first_success_result

        return {
            "success": False, "iterations": max_iterations, 
            "path": global_best, "snapshot": snapshot_path,
            "snapshot_history": self.snapshot_history,
            "disruption_iteration": disruption_iteration_recorded,
            "history": self.global_history
        }