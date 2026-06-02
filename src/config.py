"""
Constants for the experiment: algorithm parameters, experimental design, and
evaluation metrics.

One GA generation, one PSO iteration, and one ACO iteration all count as a single
iteration, even though each algorithm does a different amount of work per step.
"""

# --- Algorithm parameters ---

GA_PARAMS = {
    "selection": "roulette",
    "crossover": "two_point",
    "crossover_rate": 0.5,   # per-pair crossover probability
    "mutation_rate": 0.3,    # per-chromosome mutation probability
}

# Max path length, long enough for tortuous 40x40 paths.
GA_CHROMOSOME_LENGTH_FN = lambda w, h: 2 * (w + h)

PSO_PARAMS = {
    "omega": 1.0,   # inertia weight
    "c1": 0.1,      # cognitive coefficient
    "c2": 0.2,      # social coefficient
    "vmax": 1.0,    # velocity clamp per component; 1.0 keeps moves single-cell
}

ACO_PARAMS = {
    "alpha": 1,    # pheromone exponent
    "beta": 5,     # heuristic exponent
    "Q": 2,        # pheromone deposit amount
    "rho": 0.1,    # evaporation rate
    "tau0": 0.8,   # initial pheromone
}

# --- Experimental design ---

POPULATION_SIZES = [20, 50, 150]       # scale sweep
MAZE_TYPES = ["Shortest Path Trap", "Sudden Wall", "Parallel Paths"]
INSTANCE_SEEDS = list(range(1, 11))    # 10 distinct maze instances per cell
NUM_TRIALS = 10                        # independent trials per instance
MAZE_WIDTH = 40
MAZE_HEIGHT = 40

# --- Maze 2 disruption ---
DISRUPTION_TIME = 100                  # wall added at iteration 100

# Iterations to continue running AFTER disruption so adaptation_time is measurable.
FORCED_MIN_ITERATIONS_AFTER_DISRUPTION = 200

# --- Evaluation metrics ---
RECOVERY_THRESHOLD = 0.8               # 80% entropy recovery
ENTROPY_SAMPLE_INTERVAL = 10           # sample entropy every N iterations
ENTROPY_FLOOR_FOR_ADAPTATION = 0.1    # below this, adaptation_time returns None

# --- Runner / IO ---
ITERATION_LIMIT_MULTIPLIER = 10        # max_iterations = optimal_path_length × this
RESULTS_CSV_PATH = "results/runs.csv"
FIGURES_OUTPUT_DIR = "figures/"
