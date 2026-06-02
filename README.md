# Comparing GA, PSO, and ACO for pathfinding in 2D mazes

Group 27, Natural Computing (Radboud University).

This project compares three nature-inspired algorithms, a Genetic Algorithm (GA),
Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO), on the same
task: finding a path from start to goal in a 2D grid maze. We test them on three
maze types, across three population sizes, and measure how often they reach the
goal, how good the paths are, and how the swarm's diversity (Shannon entropy)
behaves over time. One maze type drops a wall mid-run so we can also look at how
each algorithm recovers from a sudden change.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

The dependencies are numpy, matplotlib, and tqdm, plus pytest for the tests.
They are pinned in `uv.lock`, so you get the exact versions we used.

## Setup

```bash
git clone git@github.com:manon-ru/natcomproject.git
cd natcomproject
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` and creates a virtual environment
with the locked versions. If you would rather use pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy matplotlib tqdm pytest
```

## Running the experiments

The full experiment is a 3 x 3 x 3 design (algorithm x maze type x population
size) with 10 maze instances and 10 trials per instance, so 2700 runs in total.
It is parallelized across CPU cores and writes its results to a CSV.

```bash
uv run python main.py            # full run, 2700 runs -> results/runs.csv
uv run python main.py --quick    # smoke test, 9 runs  -> results/runs_quick.csv
uv run python main.py --pilot    # small pilot, 27 runs -> results/runs_pilot.csv
uv run python main.py --workers=4   # cap the number of parallel workers
```

Start with `--quick` to check everything is wired up before launching the full
run, which takes a while.

## Looking at a single run

To watch one algorithm solve one maze and see its path and entropy curve:

```bash
uv run python scripts/visualize_run.py --algo GA  --maze "Shortest Path Trap"
uv run python scripts/visualize_run.py --algo PSO --maze "Sudden Wall" --pop 50
uv run python scripts/visualize_run.py --algo ACO --maze "Parallel Paths" --pop 150
```

Useful flags: `--small` uses a 20x20 maze instead of 40x40, `--pop` sets the
population size, `--seed` picks the maze instance, and `--no-entropy` hides the
entropy panel.

## Reproducing the figures and analysis

After a full run has produced `results/runs.csv`:

```bash
uv run python scripts/figures.py            # plots in figures/
uv run python scripts/aggregate_summary.py  # summary stats -> results/aggregate_summary.txt
uv run python scripts/hypothesis_report.py  # statistical tests -> results/hypothesis_report.txt
```

## Sample run

```
uv run python main.py --quick
Mode: quick
Total tasks: 9
Workers: 8
Output: results/runs_quick.csv
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:08<00:00,  1.01run/s]
Done. 9 results written to results/runs_quick.csv in 9.0s
```

Each CSV row is one trial: the algorithm, maze type, population size, the two
seeds, whether the goal was reached, the number of iterations, the path length,
and the entropy history encoded as JSON. Example maze previews live in
`figures/maze_previews/`.

## Layout

```
src/
  algorithms/    GA, PSO, ACO
  maze/          maze generation and the grid environment
  evaluation/    metrics, including Shannon entropy
  visualization/ plotting helpers
  config.py      experiment parameters from the proposal
  runner.py      parallel experiment runner
main.py          experiment entry point
scripts/         figures, sweeps, single-run viewer, QA checks
tests/           pytest suite
results/         experiment output (CSV and summaries)
figures/         generated plots
```

## Tests

```bash
uv run pytest
```

The suite covers the GA, PSO, and ACO operators, their invariants, and a few
small integration runs on tiny mazes.
