"""
Particle Swarm Optimization for grid-based maze navigation.

Canonical continuous PSO with grid discretization per Shi & Eberhart (1998,
ICEC Proceedings), "A modified particle swarm optimizer".

Each particle holds:
    position  ∈ ℝ²  — continuous coordinate in the maze plane
    velocity  ∈ ℝ²  — continuous step vector

VELOCITY UPDATE (Shi & Eberhart 1998 Eq. 1, inertia-weight form):
    v(t+1) = ω·v(t) + c₁·r₁·(pbest − x) + c₂·r₂·(gbest − x)

POSITION UPDATE:
    x(t+1) = x(t) + v(t+1)

DISCRETIZATION (project-specific): cell = (round(x), round(y))
The maze fitness uses Manhattan distance from the discretized cell to maze.goal,
and the path / visited trail is built from the sequence of cells the particle
walks through.

ABSORPTIVE BOUNDARY: if the proposed cell is out of bounds, lies behind a wall,
or skips more than one cell in one step (illegal in a grid), the particle is
absorbed — its velocity is zeroed and its continuous position is restored to the
last valid coordinate. Single-cell moves into adjacent open cells are accepted
and update the discrete path / visited set.

ITERATION SEMANTICS: 1 PSO iteration = 1 swarm-wide velocity+position update,
matching the GA generation and ACO iteration units used elsewhere in the runner.
"""
import numpy as np

from evaluation.metrics import calculate_shannon_entropy
from maze.environment import MazeEnvironment

# Break in the matplotlib polyline so successive segments do not zig-zag.
NAN_NODE = (float('nan'), float('nan'))


def _manhattan(c1: tuple, c2: tuple) -> int:
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])


class PSO:
    """
    Inertia-weight PSO over continuous ℝ² positions, projected onto the maze
    grid for path / entropy / collision purposes.
    """

    def __init__(
        self,
        maze: MazeEnvironment,
        num_particles: int = 50,
        omega: float = 1.0,
        c1: float = 0.1,
        c2: float = 0.2,
    ):
        self.maze = maze
        self.num_particles = num_particles
        self.omega = omega  # inertia weight (Shi & Eberhart 1998 key contribution)
        self.c1 = c1        # cognitive coefficient
        self.c2 = c2        # social coefficient

        # Swarm best — continuous position + Manhattan distance of its cell to goal.
        start_cell = self.maze.start
        self.global_best_position: np.ndarray = np.array(
            [float(start_cell[0]), float(start_cell[1])]
        )
        self.global_best_distance: float = float(_manhattan(start_cell, self.maze.goal))
        # Discrete path snapshot of whichever particle currently owns global_best.
        self.global_best_path: list[tuple] = [start_cell]

        # Telemetry consumed by runner.py / plotting layer.
        self.entropy_history: list[float] = []
        self.global_history: list[tuple] = []     # collective trail with NAN_NODE breaks
        self.snapshot_history: list[tuple] = []   # trail captured at disruption time

    # ------------------------------------------------------------------ #
    # Particle factory
    # ------------------------------------------------------------------ #
    def initialize_particles(self) -> list[dict]:
        """
        Each particle starts at maze.start with a small random velocity in
        [-1, 1]² (per Shi & Eberhart 1998 §3 — random initial velocity).
        """
        start_cell = self.maze.start
        start_pos = np.array([float(start_cell[0]), float(start_cell[1])])
        start_distance = float(_manhattan(start_cell, self.maze.goal))

        particles: list[dict] = []
        for _ in range(self.num_particles):
            particles.append(
                {
                    "position": start_pos.copy(),
                    "velocity": np.random.uniform(-1.0, 1.0, size=2),
                    "personal_best_position": start_pos.copy(),
                    "personal_best_distance": start_distance,
                    "path": [start_cell],
                    "visited": {start_cell},
                }
            )
        return particles

    # ------------------------------------------------------------------ #
    # Grid projection
    # ------------------------------------------------------------------ #
    def discretize_position(self, pos: np.ndarray) -> tuple:
        """
        Map a continuous position to its grid cell.

        Project rule: round each component to the nearest integer, so a
        continuous coordinate sitting exactly on a cell midpoint maps to the
        nearer cell. NumPy's banker's-rounding is used for numerical
        consistency across the codebase.
        """
        return (int(np.round(pos[0])), int(np.round(pos[1])))

    # ------------------------------------------------------------------ #
    # Velocity update (Shi & Eberhart 1998, Eq. 1)
    # ------------------------------------------------------------------ #
    def update_velocity(self, particle: dict) -> None:
        """
        v(t+1) = ω·v(t) + c₁·r₁·(pbest − x) + c₂·r₂·(gbest − x)

        r1, r2 are drawn from a uniform [0, 1] distribution per dimension per
        step via np.random.uniform — this keeps the algorithm controllable via
        np.random.seed() and consistent with the rest of the numpy stack.
        """
        r1 = np.random.uniform(0.0, 1.0, size=2)
        r2 = np.random.uniform(0.0, 1.0, size=2)
        particle["velocity"] = (
            self.omega * particle["velocity"]
            + self.c1 * r1 * (particle["personal_best_position"] - particle["position"])
            + self.c2 * r2 * (self.global_best_position - particle["position"])
        )

    # ------------------------------------------------------------------ #
    # Position update with absorptive boundary
    # ------------------------------------------------------------------ #
    def update_position(self, particle: dict) -> None:
        """
        Apply x(t+1) = x(t) + v(t+1), then enforce the grid constraints:

        - if the discretized cell did not change, accept the new continuous
          position unconditionally (the particle is still inside the same cell);
        - if it changed to an adjacent open cell, accept the move and extend the
          path / visited set with that cell;
        - otherwise (out-of-bounds, walled-off, or a multi-cell jump that skips
          past an unchecked wall) the particle is absorbed: continuous position
          and the discrete trail are kept as they were and velocity is zeroed.
        """
        new_pos = particle["position"] + particle["velocity"]
        prev_cell = self.discretize_position(particle["position"])
        new_cell = self.discretize_position(new_pos)

        if new_cell == prev_cell:
            particle["position"] = new_pos
            return

        step_distance = _manhattan(prev_cell, new_cell)
        adjacent_open = (
            step_distance == 1
            and self.maze.in_bounds(*new_cell)
            and not self.maze.has_wall_between(prev_cell, new_cell)
        )

        if adjacent_open:
            particle["position"] = new_pos
            if new_cell not in particle["visited"]:
                particle["path"].append(new_cell)
                particle["visited"].add(new_cell)
            return

        # Absorptive boundary: velocity zeroed, continuous position restored.
        particle["velocity"] = np.array([0.0, 0.0])

    # ------------------------------------------------------------------ #
    # Combined per-particle step
    # ------------------------------------------------------------------ #
    def update_particle(
        self,
        particle: dict,
        global_best_position: np.ndarray,
        disruption_iteration: int = -1,
    ) -> None:
        """
        Run one velocity + position update for `particle`, then refresh its
        personal best if its current cell is closer to the goal.

        `global_best_position` is accepted as an argument for the runner-style
        contract but the swarm's authoritative best lives on `self`; we keep
        both in sync via run().

        No freeze-on-goal gate here — particles continue to flow even after
        reaching the goal so the swarm keeps producing entropy samples and the
        post-disruption recovery dynamics remain observable.
        """
        # Sync external argument onto self in case a caller pre-staged it.
        self.global_best_position = global_best_position

        self.update_velocity(particle)
        self.update_position(particle)

        current_cell = particle["path"][-1]
        current_distance = float(_manhattan(current_cell, self.maze.goal))
        if current_distance < particle["personal_best_distance"]:
            particle["personal_best_position"] = particle["position"].copy()
            particle["personal_best_distance"] = current_distance

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(
        self,
        max_iterations: int = 1000,
        disruption_iteration: int = -1,
        forced_min_iterations: int = 0,
    ) -> dict:
        particles = self.initialize_particles()

        # Reset per-run telemetry / global state.
        start_cell = self.maze.start
        self.global_best_position = np.array([float(start_cell[0]), float(start_cell[1])])
        self.global_best_distance = float(_manhattan(start_cell, self.maze.goal))
        self.global_best_path = [start_cell]
        self.global_history = []
        self.snapshot_history = []
        self.entropy_history = []

        snapshot_path = None
        wall_dropped = False
        disruption_iteration_recorded = None

        # Cached result of the first successful particle, returned once
        # forced_min_iterations has also been satisfied.
        first_success_result = None

        for iteration in range(max_iterations):
            # ---- Temporal disruption: drop the dynamic wall in mid-run ----
            if (
                not wall_dropped
                and disruption_iteration > 0
                and iteration == disruption_iteration
                and getattr(self.maze, "dynamic_wall", None)
            ):
                wall_dropped = True
                snapshot_path = self.global_best_path[:]
                self.snapshot_history = self.global_history[:]
                disruption_iteration_recorded = iteration

                w1, w2 = self.maze.dynamic_wall
                self.maze.add_wall(w1, w2)

                # Truncate any discrete trail that crossed the new wall and
                # snap the continuous position back to the truncated endpoint.
                for part in particles:
                    path = part["path"]
                    for i in range(len(path) - 1):
                        if (path[i] == w1 and path[i + 1] == w2) or \
                           (path[i] == w2 and path[i + 1] == w1):
                            part["path"] = path[: i + 1]
                            part["visited"] = set(part["path"])
                            anchor = part["path"][-1]
                            part["position"] = np.array(
                                [float(anchor[0]), float(anchor[1])]
                            )
                            part["velocity"] = np.array([0.0, 0.0])
                            part["personal_best_position"] = part["position"].copy()
                            part["personal_best_distance"] = float(
                                _manhattan(anchor, self.maze.goal)
                            )
                            break

            # ---- Entropy sample every 10 iterations from each particle's cell ----
            if iteration % 10 == 0:
                cells = [p["path"][-1] for p in particles]
                self.entropy_history.append(calculate_shannon_entropy(cells))

            # ---- One swarm-wide step ----
            for particle in particles:
                prev_cell = particle["path"][-1]
                self.update_particle(particle, self.global_best_position, disruption_iteration)
                new_cell = particle["path"][-1]

                # Record the move into the collective trail for visualization.
                if new_cell != prev_cell:
                    self.global_history.append(prev_cell)
                    self.global_history.append(new_cell)
                    self.global_history.append(NAN_NODE)

                # Promote to swarm best if this particle's personal best is closer.
                if particle["personal_best_distance"] < self.global_best_distance:
                    self.global_best_distance = particle["personal_best_distance"]
                    self.global_best_position = particle["personal_best_position"].copy()
                    self.global_best_path = particle["path"][:]

                # Success: a particle has stepped onto maze.goal.
                if particle["path"][-1] == self.maze.goal:
                    # Ignore pre-disruption successes when running a Sudden-Wall trial.
                    if disruption_iteration > 0 and iteration < disruption_iteration:
                        continue
                    if first_success_result is None:
                        first_success_result = {
                            "success": True,
                            "iterations": iteration + 1,
                            "path": particle["path"][:],
                            "snapshot": snapshot_path,
                            "snapshot_history": self.snapshot_history,
                            "disruption_iteration": disruption_iteration_recorded,
                        }

            # Only honor success once forced_min_iterations has elapsed.
            if first_success_result is not None and iteration >= forced_min_iterations:
                first_success_result["history"] = self.global_history
                return first_success_result

        # Loop exhausted — return cached success if we have one, else best-so-far.
        if first_success_result is not None:
            first_success_result["history"] = self.global_history
            return first_success_result

        return {
            "success": False,
            "iterations": max_iterations,
            "path": self.global_best_path,
            "snapshot": snapshot_path,
            "snapshot_history": self.snapshot_history,
            "disruption_iteration": disruption_iteration_recorded,
            "history": self.global_history,
        }
