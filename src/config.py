"""
config.py — Proposal-aligned constants for Group 27 NatComp experiment.

All values trace to proposal.txt:
  - Section 4.1: algorithm parameters
  - Section 4.3: experimental design (population sizes, maze types, instance/trial counts)
  - Section 4.4: evaluation metrics (recovery threshold, entropy floor)

ITERATION SEMANTICS:
1 GA generation = 1 PSO iteration = 1 ACO iteration.
This is the algorithm-native iteration unit. Different algorithms perform different
amounts of work per iteration; this is documented in the report's Limitations section.
"""

# --- Algorithm Parameters (proposal Section 4.1) ---

GA_PARAMS = {
    "selection": "roulette",
    "crossover": "two_point",
    "crossover_rate": 0.5,   # "split 0.5" = per-pair crossover probability 0.5
    "mutation_rate": 0.3,    # flip-bit rate 0.3 = per-gene replacement probability
}

# Chromosome length = 2*(W+H). Sufficient for tortuous 40×40 paths; configurable.
GA_CHROMOSOME_LENGTH_FN = lambda w, h: 2 * (w + h)

PSO_PARAMS = {
    "omega": 1.0,   # inertia weight (proposal Section 4.1)
    "c1": 0.1,      # cognitive coefficient (proposal Section 4.1)
    "c2": 0.2,      # social coefficient (proposal Section 4.1)
    "vmax": 1.0,    # velocity clamp per component; 1.0 keeps moves single-cell
}

ACO_PARAMS = {
    "alpha": 1,    # pheromone exponent
    "beta": 5,     # heuristic exponent (was 2.0 in code — corrected)
    "Q": 2,        # pheromone deposit amount (was 1.0 in code — corrected)
    "rho": 0.1,    # evaporation rate
    "tau0": 0.8,   # initial pheromone (was 0.1 in code — corrected)
}

# --- Experimental Design (proposal Section 4.3) ---

POPULATION_SIZES = [20, 50, 150]       # scale sweep
MAZE_TYPES = ["Shortest Path Trap", "Sudden Wall", "Parallel Paths"]
INSTANCE_SEEDS = list(range(1, 11))    # 10 distinct maze instances per cell
NUM_TRIALS = 10                        # independent trials per instance
MAZE_WIDTH = 40
MAZE_HEIGHT = 40

# --- Maze 2 Disruption (proposal Section 4.3) ---
DISRUPTION_TIME = 100                  # wall added at iteration 100 (was 150 — corrected)

# Iterations to continue running AFTER disruption so adaptation_time is measurable.
FORCED_MIN_ITERATIONS_AFTER_DISRUPTION = 200

# --- Evaluation Metrics (proposal Section 4.4) ---
RECOVERY_THRESHOLD = 0.8               # 80% entropy recovery (was 0.5 — corrected)
ENTROPY_SAMPLE_INTERVAL = 10           # sample entropy every N iterations
ENTROPY_FLOOR_FOR_ADAPTATION = 0.1    # below this, adaptation_time returns None

# --- Runner / IO ---
ITERATION_LIMIT_MULTIPLIER = 10        # max_iterations = optimal_path_length × this
RESULTS_CSV_PATH = "results/runs.csv"
FIGURES_OUTPUT_DIR = "figures/"
