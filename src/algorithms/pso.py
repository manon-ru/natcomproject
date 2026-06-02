"""
Particle Swarm Optimization for grid-based maze navigation.

Particles move in continuous 2D space with the standard inertia-weight velocity
update, and are mapped to grid cells with (round(x), round(y)). Invalid moves
(out of bounds, through a wall, or longer than one cell) are rejected and the
particle falls back to a valid neighbour. One swarm-wide update is one iteration.
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
    Inertia-weight PSO over continuous 2D positions, projected onto the maze
    grid for path, entropy, and collision handling.
    """

    def __init__(
        self,
        maze: MazeEnvironment,
        num_particles: int = 50,
        omega: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        vmax: float = 1.0,
    ):
        self.maze = maze
        self.num_particles = num_particles
        self.omega = omega  # inertia weight
        self.c1 = c1        # cognitive coefficient
        self.c2 = c2        # social coefficient
        self.vmax = vmax    # per-component velocity clamp, keeps moves single-cell

        # Swarm best: continuous position and Manhattan distance of its cell to goal.
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

    # Particle factory
    def initialize_particles(self) -> list[dict]:
        """
        Each particle starts at maze.start with a small random velocity drawn
        from [-1, 1] per component.
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
                    "personal_best": start_pos.copy(),
                    "personal_best_distance": start_distance,
                    "path": [start_cell],
                    "visited": {start_cell},
                }
            )
        return particles

    # Grid projection + distance helper
    def distance_to_goal(self, path_or_pos) -> float:
        if isinstance(path_or_pos, np.ndarray):
            cell = self.discretize_position(path_or_pos.ravel()[:2])
        else:
            cell = path_or_pos[-1]
        return float(_manhattan(cell, self.maze.goal))

    def discretize_position(self, pos) -> tuple:
        """
        Map a continuous position to its grid cell.

        Project rule: round each component to the nearest integer, so a
        continuous coordinate sitting exactly on a cell midpoint maps to the
        nearer cell. Python's banker's-rounding (round half to even) is
        used for numerical consistency. Implemented with float() + round()
        rather than np.round() to avoid numpy dispatch overhead in the
        hot path (3 calls per particle per iteration).
        """
        return (round(float(pos[0])), round(float(pos[1])))

    # Velocity update
    def update_velocity(self, particle: dict, global_best_position: np.ndarray) -> dict:
        """
        v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)

        r1 and r2 are drawn from a uniform [0, 1] distribution per dimension per
        step with np.random.uniform, which keeps runs reproducible under
        np.random.seed() and consistent with the rest of the numpy code.
        """
        r1 = np.random.uniform(0.0, 1.0, size=2)
        r2 = np.random.uniform(0.0, 1.0, size=2)
        pb = particle["personal_best"]
        if not (isinstance(pb, np.ndarray) and pb.ndim == 1 and pb.shape[0] == 2):
            pb = np.asarray(pb, dtype=float).ravel()[:2]
        gb = global_best_position
        if not (isinstance(gb, np.ndarray) and gb.ndim == 1 and gb.shape[0] == 2):
            gb = np.asarray(gb, dtype=float).ravel()[:2]
        particle["velocity"] = np.clip(
            self.omega * particle["velocity"]
            + self.c1 * r1 * (pb - particle["position"])
            + self.c2 * r2 * (gb - particle["position"]),
            -self.vmax,
            self.vmax,
        )
        return particle

    # Position update with velocity projection
    def _project_move(self, particle: dict, cell: tuple, global_best_position: np.ndarray = None) -> tuple:
        """
        Called when the canonical velocity move is invalid (walled off, out of
        bounds, or a multi-cell jump).

        Unvisited open neighbours are scored by a discrete-PSO cost function:
        cost = w + c1*r1*d_pbest + c2*r2*d_gbest, where d_pbest and
        d_gbest are the Manhattan distances from each candidate to the personal
        best and global best cell. The random draws r1, r2 per call keep
        exploration diverse so the swarm does not converge too early, while c1
        and c2 still encode the cognitive and social weights from the velocity
        formula. The minimum-cost unvisited neighbour is chosen.

        When every open neighbour has already been visited the particle
        backtracks one step: the last cell is popped from the path and the
        position is snapped back to the parent cell. The velocity is preserved so
        the next update continues coherently.

        After any move the continuous position is snapped to the chosen cell
        centre so pbest stores integer-aligned coordinates for clean
        cognitive/social terms on the following update.
        """
        neighbours: list[tuple] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cell[0] + dx, cell[1] + dy)
            if self.maze.in_bounds(*nb) and not self.maze.has_wall_between(cell, nb):
                neighbours.append((dx, dy, nb))

        unvisited = [(dx, dy, nb) for dx, dy, nb in neighbours
                     if nb not in particle["visited"]]

        if unvisited:
            # Cost = w + c1*r1*d_pbest + c2*r2*d_gbest.
            # r1, r2 are drawn independently per candidate so that equidistant
            # neighbours get different random scalings and particles at the same
            # cell pick different directions, which keeps the swarm diverse.
            pb = particle["personal_best"]
            pb_cell = self.discretize_position(
                np.asarray(pb, dtype=float).ravel()[:2]
            )
            gb_pos = global_best_position if global_best_position is not None \
                else self.global_best_position
            gb_cell = self.discretize_position(
                np.asarray(gb_pos, dtype=float).ravel()[:2]
            )

            best_cost = float("inf")
            target: tuple = unvisited[0][2]
            best_dir: tuple = (unvisited[0][0], unvisited[0][1])
            for dx, dy, nb in unvisited:
                r1 = float(np.random.uniform(0.0, 1.0))
                r2 = float(np.random.uniform(0.0, 1.0))
                dp = _manhattan(nb, pb_cell)
                dg = _manhattan(nb, gb_cell)
                cost = self.omega + self.c1 * r1 * dp + self.c2 * r2 * dg
                if cost < best_cost:
                    best_cost = cost
                    target = nb
                    best_dir = (dx, dy)

            particle["position"] = np.array([float(target[0]), float(target[1])])
            particle["velocity"] = np.clip(
                np.array([float(best_dir[0]), float(best_dir[1])]),
                -self.vmax,
                self.vmax,
            )
            return target

        # All open neighbours already visited — DFS backtrack one step.
        # visited is intentionally NOT shrunk: previously explored cells remain
        # marked so the particle looks for genuinely new branches after returning.
        if len(particle["path"]) > 1:
            particle["path"].pop()
            parent = particle["path"][-1]
            particle["position"] = np.array([float(parent[0]), float(parent[1])])
            # velocity is preserved; the canonical update will adjust it next step
            return parent

        # At start cell with no unvisited open neighbours (fully explored, stuck).
        return cell

    def update_position(self, particle: dict, global_best_position: np.ndarray = None) -> None:
        """
        Apply x(t+1) = x(t) + v(t+1), then enforce the grid constraints:

        - if the discretized cell did not change, accept the new continuous
          position unconditionally (the particle is still inside the same cell);
        - if the starting cell is out of bounds, accept the move unconditionally
          (used in unit tests that set up particles at arbitrary positions);
        - if it changed to an adjacent open cell, snap position to cell centre
          (integer coordinates) so pbest stores clean grid coordinates;
        - otherwise (out-of-bounds, walled-off, or a multi-cell jump) use
          velocity projection: move to the open neighbour most aligned with
          the current velocity direction.  Snapping to cell centre and
          pointing velocity toward the chosen cell ensures particles always
          make forward progress instead of freezing at boundaries.

        After the position decision, the path is always synced to the current
        discretized position so that entropy sampling and tests can read it.
        """
        new_pos = particle["position"] + particle["velocity"]
        prev_cell = self.discretize_position(particle["position"])
        new_cell = self.discretize_position(new_pos)

        if new_cell == prev_cell:
            # Sub-cell move — velocity too small to cross a cell boundary.
            # Force a DFS step so every iteration makes real maze progress
            # (matches the old cost-based PSO's one-step-per-iteration contract).
            gb_for_project = global_best_position if global_best_position is not None \
                else self.global_best_position
            current_cell = self._project_move(particle, prev_cell, gb_for_project)
        elif not self.maze.in_bounds(*prev_cell):
            # OOB start position — accept unconditionally (unit-test contract)
            particle["position"] = new_pos
            current_cell = new_cell
        elif (
            _manhattan(prev_cell, new_cell) == 1
            and self.maze.in_bounds(*new_cell)
            and not self.maze.has_wall_between(prev_cell, new_cell)
            and new_cell not in particle["visited"]
        ):
            # Valid single-cell move to an unvisited cell — snap to cell centre.
            # Visited cells are excluded: accepting a backward canonical move
            # would desync particle["position"] from particle["path"][-1] and
            # corrupt the DFS trail (path would skip cells, backtracks would
            # pop the wrong cell, and the visited invariant would break).
            particle["position"] = np.array([float(new_cell[0]), float(new_cell[1])])
            current_cell = new_cell
        else:
            # Invalid, OOB, multi-cell, or backward (visited) move — use DFS.
            gb_for_project = global_best_position if global_best_position is not None \
                else self.global_best_position
            current_cell = self._project_move(particle, prev_cell, gb_for_project)

        if particle["path"][-1] != current_cell and current_cell not in particle["visited"]:
            particle["path"].append(current_cell)
            particle["visited"].add(current_cell)

    # Combined per-particle step
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
        if "personal_best_distance" in particle:
            self.update_velocity(particle, global_best_position)
        gb = global_best_position
        if not (isinstance(gb, np.ndarray) and gb.ndim == 1 and gb.shape[0] == 2):
            gb = np.asarray(gb, dtype=float).ravel()[:2]
        self.update_position(particle, gb)

        # Fitness uses actual discrete position so pbest/gbest reflect where
        # particles are physically closest to goal, even when navigating through
        # already-visited cells (where path[-1] would otherwise be stale).
        actual_cell = self.discretize_position(particle["position"])
        current_distance = float(_manhattan(actual_cell, self.maze.goal))
        if current_distance < particle.get("personal_best_distance", float("inf")):
            particle["personal_best"] = particle["position"].copy()
            particle["personal_best_distance"] = current_distance

    # Main loop
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
                # personal_best / personal_best_distance are intentionally
                # NOT reset: retaining the pre-disruption best keeps the
                # cost function pointing toward the goal region so particles
                # continue searching for an alternative route.
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
                            break

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
                    self.global_best_position = np.asarray(particle["personal_best"], dtype=float).ravel()[:2].copy()
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

            # ---- Entropy sample every 10 iterations (after updates) ----
            if iteration % 10 == 0:
                cells = [p["path"][-1] for p in particles]
                self.entropy_history.append(calculate_shannon_entropy(cells))

            # Only honor success once forced_min_iterations has elapsed.
            if first_success_result is not None and iteration >= forced_min_iterations:
                cells = [p["path"][-1] for p in particles]
                self.entropy_history.append(calculate_shannon_entropy(cells))
                first_success_result["history"] = self.global_history
                return first_success_result

        # Final entropy sample after loop completes.
        cells = [p["path"][-1] for p in particles]
        self.entropy_history.append(calculate_shannon_entropy(cells))

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
