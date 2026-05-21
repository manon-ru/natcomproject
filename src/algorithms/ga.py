"""
Genetic Algorithm for grid-based maze navigation.

Chromosome encoding: VARIABLE-LENGTH SEQUENCE OF CELLS visited from start.
Following Lamini et al. (2018, Procedia Computer Science) and Tu & Yang (2003, IEEE ICRA),
each chromosome is a list of (x, y) tuples representing the cells traversed from
maze.start, with maze.goal appearing in the path indicating success.

This differs from Shrestha et al.'s 36-bit waypoint encoding, which suits continuous
2D environments with random waypoints scattered through free space. Our grid-based
formulation with cell-by-cell maze navigation has no continuous space between cells —
the path is the chromosome's natural representation.

OPERATORS (proposal Section 4.1, adapted to path encoding):
- Roulette wheel selection over shifted fitness = -manhattan_to_goal (or -path_length if goal reached)
- Two-point crossover at common cells (splice at two cells appearing in both parents)
- Per-chromosome mutation at rate 0.3 = truncate at random cell and regrow via random walk
  (For variable-length path representations, per-element flip-bit mutation has no
  well-defined analogue because changing one cell breaks neighbor adjacency. We adopt
  the standard adaptation: a per-chromosome mutation probability with a truncate-regrow
  operation, as in Lamini et al.)

ITERATION SEMANTICS: 1 GA generation = 1 PSO/ACO iteration (algorithm-native unit).
"""
import random
from evaluation.metrics import calculate_shannon_entropy
from maze.environment import MazeEnvironment

NAN_NODE = (float('nan'), float('nan'))


class GeneticAlgorithm:
    def __init__(
        self,
        maze: MazeEnvironment,
        pop_size: int = 50,
        chromosome_length: int = None,   # max path length (kept for API compat with runner.py)
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.3,
    ):
        self.maze = maze
        self.pop_size = pop_size
        self.max_path_length = chromosome_length if chromosome_length is not None else 2 * (maze.width + maze.height)
        self.chromosome_length = self.max_path_length   # alias for runner.py compatibility
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.entropy_history: list[float] = []
        self.global_history: list[tuple] = []
        self.snapshot_history: list[tuple] = []

    # ── Path construction ────────────────────────────────────────────────────

    def _random_walk(self, start_path: list[tuple]) -> list[tuple]:
        """Extend a path by random unvisited-neighbor walk until stuck, goal, or max length."""
        path = list(start_path)
        visited = set(path)
        while len(path) < self.max_path_length:
            curr = path[-1]
            if curr == self.maze.goal:
                break
            neighbors = self.maze.neighbors(*curr)
            options = [n for n in neighbors if n not in visited]
            if not options:
                break
            next_cell = random.choice(options)
            path.append(next_cell)
            visited.add(next_cell)
        return path

    def _initialize_population(self) -> list[dict]:
        return [{"path": self._random_walk([self.maze.start])} for _ in range(self.pop_size)]

    def _truncate_at_goal(self, path: list[tuple]) -> list[tuple]:
        """Return prefix of path up to and including first goal occurrence (or whole path)."""
        if self.maze.goal in path:
            idx = path.index(self.maze.goal)
            return path[:idx + 1]
        return path

    # ── Fitness & entropy ────────────────────────────────────────────────────

    def _reached_goal(self, ind: dict) -> bool:
        return self.maze.goal in ind["path"]

    def _fitness(self, ind: dict) -> float:
        path = ind["path"]
        if self.maze.goal in path:
            # Shorter goal-reaching paths score higher
            goal_idx = path.index(self.maze.goal)
            return -goal_idx + self.max_path_length
        end = path[-1]
        gx, gy = self.maze.goal
        return -(abs(end[0] - gx) + abs(end[1] - gy))

    def _entropy_position(self, ind: dict) -> tuple:
        """Goal if reached, else end position (per Metis Q2(b) policy)."""
        return self.maze.goal if self._reached_goal(ind) else ind["path"][-1]

    # ── Operators ────────────────────────────────────────────────────────────

    def _roulette_select(self, population: list[dict]) -> dict:
        fitnesses = [self._fitness(ind) for ind in population]
        min_f = min(fitnesses)
        shifted = [f - min_f + 1e-9 for f in fitnesses]
        total = sum(shifted)
        pick = random.uniform(0, total)
        cum = 0.0
        for ind, s in zip(population, shifted):
            cum += s
            if cum >= pick:
                return ind
        return population[-1]

    def _crossover(self, parent_a: dict, parent_b: dict) -> dict:
        """Two-point crossover at common cells. With probability crossover_rate, splice."""
        if random.random() >= self.crossover_rate:
            return {"path": parent_a["path"][:]}

        path_a = parent_a["path"]
        path_b = parent_b["path"]
        set_b = set(path_b)
        # Common cells appearing in both parents (excluding the shared start cell)
        common = [c for c in path_a if c in set_b and c != self.maze.start]
        if not common:
            # Adjacency-aware splice: find (A_i, B_j) where B_j ∈ maze.neighbors(*A_i) and B_j ∈ path_b
            candidates = []
            for idx_a in range(1, len(path_a)):
                a_cell = path_a[idx_a]
                for b_cell in self.maze.neighbors(*a_cell):
                    if b_cell in set_b:
                        candidates.append((idx_a, b_cell))
            if candidates:
                idx_a, b_cell = random.choice(candidates)
                idx_b = path_b.index(b_cell)
                child_path = path_a[:idx_a + 1] + path_b[idx_b:]
            else:
                # Tertiary fallback: fresh random walk from start
                child_path = self._random_walk([self.maze.start])
        elif len(common) >= 2:
            # Two-point: pick two common cells, sorted by their position in parent A
            picks = random.sample(common, 2)
            picks.sort(key=lambda c: path_a.index(c))
            c1, c2 = picks
            idx_a1, idx_a2 = path_a.index(c1), path_a.index(c2)
            idx_b1, idx_b2 = path_b.index(c1), path_b.index(c2)
            # Make sure idx_b1 < idx_b2 in parent B for a valid splice
            if idx_b1 < idx_b2:
                child_path = path_a[:idx_a1] + path_b[idx_b1:idx_b2] + path_a[idx_a2:]
            else:
                # parent B traverses c2 before c1 — fall back to single-point at c1
                child_path = path_a[:idx_a1] + path_b[idx_b1:]
        else:
            # Single common cell — one-point splice
            c = common[0]
            idx_a = path_a.index(c)
            idx_b = path_b.index(c)
            child_path = path_a[:idx_a] + path_b[idx_b:]

        # Clean revisits: if a cell appears twice, truncate at its first occurrence.
        # Design choice (intentional, not a bug): when the splice creates a loop — a cell
        # appearing in both the path_b segment and the path_a tail — we drop the loop body
        # by truncating at the first occurrence. Adjacency is preserved because the cell
        # immediately after the loop was originally adjacent to the loop's entry cell in the
        # parent path. Loops are strictly wasteful in pathfinding (extra cells, no progress),
        # so removing them is a free path-length reduction. See Lamini et al. (2018).
        seen = set()
        clean = []
        for cell in child_path:
            if cell in seen:
                idx = clean.index(cell)
                clean = clean[:idx + 1]
                seen = set(clean)
            else:
                clean.append(cell)
                seen.add(cell)

        # Cap at max length
        if len(clean) > self.max_path_length:
            clean = clean[:self.max_path_length]

        return {"path": clean}

    def _mutate(self, ind: dict) -> dict:
        """Per-chromosome mutation at rate mutation_rate: truncate at random cell and regrow."""
        if random.random() >= self.mutation_rate:
            return ind
        path = ind["path"]
        if len(path) <= 2:
            return ind
        new_path = path
        for _ in range(3):
            truncate_at = random.randint(1, len(path) - 1)
            truncated = path[:truncate_at]
            candidate = self._random_walk(truncated)
            if len(candidate) > len(truncated):
                return {"path": candidate}
            new_path = candidate
        return {"path": new_path}

    # ── Run loop ─────────────────────────────────────────────────────────────

    def run(
        self,
        max_iterations: int = 1000,
        disruption_iteration: int = -1,
        forced_min_iterations: int = 0,
    ) -> dict:
        population = self._initialize_population()
        self.entropy_history = []
        self.global_history = []
        self.snapshot_history = []

        snapshot_path = None
        wall_dropped = False
        disruption_iteration_recorded = None
        first_success_result = None

        for iteration in range(max_iterations):
            # ── Disruption ───────────────────────────────────────────────────
            if not wall_dropped and disruption_iteration > 0 and iteration == disruption_iteration:
                if hasattr(self.maze, "dynamic_wall") and self.maze.dynamic_wall:
                    wall_dropped = True
                    best = max(population, key=self._fitness)
                    snapshot_path = best["path"][:]
                    self.snapshot_history = self.global_history[:]
                    disruption_iteration_recorded = iteration
                    w1, w2 = self.maze.dynamic_wall
                    self.maze.add_wall(w1, w2)
                    # Truncate any individual whose path crosses the new wall
                    for ind in population:
                        p = ind["path"]
                        for i in range(len(p) - 1):
                            if (p[i] == w1 and p[i + 1] == w2) or (p[i] == w2 and p[i + 1] == w1):
                                ind["path"] = p[:i + 1]
                                break

            # ── Entropy sample ───────────────────────────────────────────────
            if iteration % 10 == 0:
                positions = [self._entropy_position(ind) for ind in population]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            # ── Next generation ──────────────────────────────────────────────
            new_population = []
            for _ in range(self.pop_size):
                a = self._roulette_select(population)
                b = self._roulette_select(population)
                child = self._crossover(a, b)
                child = self._mutate(child)
                new_population.append(child)
            population = new_population

            # ── Visualization trail: best individual's path ──────────────────
            best = max(population, key=self._fitness)
            if best["path"] and len(best["path"]) > 1:
                for j in range(len(best["path"]) - 1):
                    self.global_history.append(best["path"][j])
                    self.global_history.append(best["path"][j + 1])
                    self.global_history.append(NAN_NODE)

            # ── Success check ────────────────────────────────────────────────
            for ind in population:
                if self._reached_goal(ind):
                    if disruption_iteration > 0 and iteration < disruption_iteration:
                        continue
                    if first_success_result is None:
                        path = self._truncate_at_goal(ind["path"])
                        first_success_result = {
                            "success": True,
                            "iterations": iteration + 1,
                            "path": path[:],
                            "snapshot": snapshot_path,
                            "snapshot_history": self.snapshot_history,
                            "disruption_iteration": disruption_iteration_recorded,
                        }

            if first_success_result is not None and iteration >= forced_min_iterations:
                first_success_result["history"] = self.global_history
                return first_success_result

        if first_success_result is not None:
            first_success_result["history"] = self.global_history
            return first_success_result

        best = max(population, key=self._fitness)
        return {
            "success": False,
            "iterations": max_iterations,
            "path": best["path"] or [],
            "snapshot": snapshot_path,
            "snapshot_history": self.snapshot_history,
            "disruption_iteration": disruption_iteration_recorded,
            "history": self.global_history,
        }
