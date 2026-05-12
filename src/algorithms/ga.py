"""
Genetic Algorithm for grid-based maze navigation.
Chromosome: fixed-length list of direction genes {0=U, 1=D, 2=L, 3=R}.

ITERATION SEMANTICS: 1 GA generation = 1 PSO/ACO iteration (algorithm-native unit).
Different computational work per iteration; see report Limitations section.

OPERATORS (proposal Section 4.1):
- Selection: roulette wheel over shifted fitness
- Crossover: two-point with per-pair probability crossover_rate=0.5
- Mutation: per-gene replacement probability mutation_rate=0.3
- "split 0.5" = crossover probability 0.5 (standard Shrestha-style reading)
"""
import random
from evaluation.metrics import calculate_shannon_entropy
from maze.environment import MazeEnvironment

NAN_NODE = (float('nan'), float('nan'))
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]   # U, D, L, R


class GeneticAlgorithm:
    def __init__(
        self, 
        maze: MazeEnvironment, 
        pop_size: int = 50, 
        chromosome_length: int = None,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.3,
    ):
        self.maze = maze
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length if chromosome_length is not None else 2 * (maze.width + maze.height)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.entropy_history: list[float] = []
        self.global_history: list[tuple] = []
        self.snapshot_history: list[tuple] = []

    # ── Chromosome execution ─────────────────────────────────────────────────

    def _execute_chromosome(self, chromosome: list[int]) -> dict:
        """Execute chromosome from start, stopping at goal if reached."""
        pos = self.maze.start
        path = [pos]
        reached_goal = False
        for gene in chromosome:
            dx, dy = DIRECTIONS[gene]
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.maze.in_bounds(nx, ny) and not self.maze.has_wall_between(pos, (nx, ny)):
                pos = (nx, ny)
                path.append(pos)
            if pos == self.maze.goal:
                reached_goal = True
                break
        return {"path": path, "end_position": pos, "reached_goal": reached_goal}

    def _execute_individual(self, ind: dict) -> None:
        """Re-execute an individual's chromosome against the current maze state."""
        result = self._execute_chromosome(ind["chromosome"])
        ind["executed_path"] = result["path"]
        ind["end_position"] = result["end_position"]
        ind["reached_goal"] = result["reached_goal"]

    # ── Entropy position ─────────────────────────────────────────────────────

    def _entropy_position(self, ind: dict) -> tuple:
        """Return goal if reached, else end_position. Locked per Metis Q2(b)."""
        return self.maze.goal if ind["reached_goal"] else ind["end_position"]

    # ── Fitness ──────────────────────────────────────────────────────────────

    def _fitness(self, ind: dict) -> float:
        if ind["reached_goal"]:
            # Reward shorter paths; chromosome_length is a known upper bound
            return -(len(ind["executed_path"]) - 1) + self.chromosome_length
        gx, gy = self.maze.goal
        x, y = ind["end_position"]
        return -abs(x - gx) - abs(y - gy)

    # ── Operators ────────────────────────────────────────────────────────────

    def _roulette_select(self, population: list[dict]) -> dict:
        fitnesses = [self._fitness(ind) for ind in population]
        min_f = min(fitnesses)
        shifted = [f - min_f + 1e-9 for f in fitnesses]
        total = sum(shifted)
        pick = random.uniform(0, total)
        cumulative = 0.0
        for ind, s in zip(population, shifted):
            cumulative += s
            if cumulative >= pick:
                return ind
        return population[-1]

    def _crossover(self, parent_a: dict, parent_b: dict) -> dict:
        chrom_a = parent_a["chromosome"]
        chrom_b = parent_b["chromosome"]
        L = self.chromosome_length
        if random.random() < self.crossover_rate:
            p1 = random.randint(0, L - 1)
            p2 = random.randint(p1, L)
            child_chrom = chrom_a[:p1] + chrom_b[p1:p2] + chrom_a[p2:]
        else:
            child_chrom = chrom_a[:]
        return {
            "chromosome": child_chrom,
            "executed_path": None,
            "end_position": None,
            "reached_goal": False,
        }

    def _mutate(self, ind: dict) -> dict:
        chrom = ind["chromosome"]
        for i in range(len(chrom)):
            if random.random() < self.mutation_rate:
                # Replace with a random different direction
                old = chrom[i]
                choices = [g for g in range(4) if g != old]
                chrom[i] = random.choice(choices)
        return ind

    # ── Population init ──────────────────────────────────────────────────────

    def _initialize_population(self) -> list[dict]:
        population = []
        for _ in range(self.pop_size):
            chrom = [random.randint(0, 3) for _ in range(self.chromosome_length)]
            ind = {
                "chromosome": chrom,
                "executed_path": None,
                "end_position": None,
                "reached_goal": False,
            }
            self._execute_individual(ind)
            population.append(ind)
        return population

    # ── Main run loop ────────────────────────────────────────────────────────

    def run(
        self, 
        max_iterations: int = 1000, 
        disruption_iteration: int = -1, 
        forced_min_iterations: int = 0
    ) -> dict:
        population = self._initialize_population()
        self.global_history = []
        self.entropy_history = []
        self.snapshot_history = []

        snapshot_path = None
        wall_dropped = False
        disruption_iteration_recorded = None
        first_success_result = None

        for iteration in range(max_iterations):

            # ── Temporal disruption ──────────────────────────────────────────
            if not wall_dropped and disruption_iteration > 0 and iteration == disruption_iteration:
                if hasattr(self.maze, 'dynamic_wall') and self.maze.dynamic_wall:
                    wall_dropped = True
                    # Snapshot best chromosome's path before wall drops
                    best = max(population, key=self._fitness)
                    snapshot_path = best["executed_path"][:] if best["executed_path"] else []
                    self.snapshot_history = self.global_history[:]
                    disruption_iteration_recorded = iteration
                    self.maze.add_wall(*self.maze.dynamic_wall)
                    # Re-execute all chromosomes against new maze
                    for ind in population:
                        self._execute_individual(ind)

            # ── Entropy sample ───────────────────────────────────────────────
            if iteration % 10 == 0:
                positions = [self._entropy_position(ind) for ind in population]
                self.entropy_history.append(calculate_shannon_entropy(positions))

            # ── Evolve next generation ───────────────────────────────────────
            new_population = []
            for _ in range(self.pop_size):
                parent_a = self._roulette_select(population)
                parent_b = self._roulette_select(population)
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child)
                self._execute_individual(child)
                new_population.append(child)
            population = new_population

            # ── Track best individual's path in global_history for visualization
            best = max(population, key=self._fitness)
            if best["executed_path"] and len(best["executed_path"]) > 1:
                for j in range(len(best["executed_path"]) - 1):
                    self.global_history.append(best["executed_path"][j])
                    self.global_history.append(best["executed_path"][j + 1])
                    self.global_history.append(NAN_NODE)

            # ── Success check ────────────────────────────────────────────────
            for ind in population:
                if ind["reached_goal"]:
                    if disruption_iteration > 0 and iteration < disruption_iteration:
                        continue
                    if first_success_result is None:
                        first_success_result = {
                            "success": True,
                            "iterations": iteration + 1,
                            "path": ind["executed_path"][:],
                            "snapshot": snapshot_path,
                            "snapshot_history": self.snapshot_history,
                            "disruption_iteration": disruption_iteration_recorded,
                        }

            # ── Early exit gate ──────────────────────────────────────────────
            if first_success_result is not None and iteration >= forced_min_iterations:
                first_success_result["history"] = self.global_history
                return first_success_result

        # Fallback
        if first_success_result is not None:
            first_success_result["history"] = self.global_history
            return first_success_result

        best = max(population, key=self._fitness)
        return {
            "success": False,
            "iterations": max_iterations,
            "path": best["executed_path"] or [],
            "snapshot": snapshot_path,
            "snapshot_history": self.snapshot_history,
            "disruption_iteration": disruption_iteration_recorded,
            "history": self.global_history,
        }
