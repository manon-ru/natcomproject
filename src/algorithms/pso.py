from evaluation.entropy import calculate_shannon_entropy
from maze.environment import MazeEnvironment


class PSO:
    """
    Particle Swarm Optimization for maze navigation.

    Each particle maintains a current path and a personal best.
    The swarm shares a global best. Particles update their trajectories
    based on personal and swarm bests.
    """

    def __init__(self, maze: MazeEnvironment, num_particles: int = 30):
        self.maze = maze
        self.num_particles = num_particles
        self.entropy_history: list[float] = []

    def _initialize_particles(self) -> list[dict]:
        return [
            {"path": [self.maze.start], "personal_best": [self.maze.start]}
            for _ in range(self.num_particles)
        ]

    def _distance_to_goal(self, path: list[tuple]) -> float:
        # TODO: return a scalar distance from the path head to the goal
        raise NotImplementedError

    def _update_particle(self, particle: dict, global_best: list[tuple]) -> dict:
        # TODO: move particle one step, biased toward personal_best and global_best;
        #       update personal_best if the new position is better
        raise NotImplementedError

    def run(self, max_iterations: int = 1000) -> dict:
        particles = self._initialize_particles()
        global_best: list[tuple] = [self.maze.start]

        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                positions = [p["path"][-1] for p in particles]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            for particle in particles:
                particle = self._update_particle(particle, global_best)

                if self._distance_to_goal(particle["path"]) < self._distance_to_goal(global_best):
                    global_best = particle["path"][:]

                if particle["path"][-1] == self.maze.goal:
                    return {"success": True, "iterations": iteration + 1, "path": particle["path"]}

        return {"success": False, "iterations": max_iterations, "path": None}
