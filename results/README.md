# Results Directory

## File Suffix Conventions

| Suffix | Meaning | When Created |
|--------|---------|--------------|
| (none) | Current canonical state | Regenerated after each major fix or full experiment run |
| `.pre-fix` | Snapshot before GA crossover/mutation correctness fix | Before GA operator fixes (earlier in project) |
| `.pre-canonical` | Snapshot before PSO/ACO canonicalization | Before PSO/ACO rewrite to canonical formulations |

Never delete or overwrite `.pre-fix` or `.pre-canonical` files. They are the historical record.

## SHA Reference Table

| File | Tag / SHA | Description |
|------|-----------|-------------|
| `runs.csv` | `<filled-in-by-T23>` | Full 2700-run experiment results (canonical PSO/ACO) |
| `runs.csv.pre-canonical` | `v-pre-pso-aco-canonical` = `402997c` | Pre-PSO/ACO-canonicalization data |
| `runs.csv.pre-fix` | (see git log) | Pre-GA-operator-fix data |
| `hypothesis_report.txt` | `<filled-in-by-T23>` | Hypothesis test results (canonical) |
| `aggregate_summary.txt` | `<filled-in-by-T23>` | Aggregate statistics (canonical) |

Tag `v-pre-pso-aco-canonical` points to commit `402997ce2d8e2330fd137aea4e9588e2419133a7`.

## Experimental Design

3 algorithms × 3 maze types × 3 population sizes × 10 instances × 10 trials = **2700 runs**

- **Algorithms**: GA, PSO, ACO
- **Maze types**: Shortest Path Trap, Sudden Wall, Parallel Paths
- **Population sizes**: 20, 50, 150
- **Instances per cell**: 10 (different random seeds)
- **Trials per instance**: 10 (different algorithm seeds)

## CSV Column Schema

Each row in `runs.csv` corresponds to one trial. Columns:

| Column | Type | Description |
|--------|------|-------------|
| `algo` | str | Algorithm name: GA, PSO, or ACO |
| `maze_type` | str | Maze type string |
| `pop_size` | int | Population / swarm / colony size |
| `instance_seed` | int | Seed used to generate the maze instance |
| `trial_seed` | int | Seed used for algorithm randomness |
| `success_overall` | bool | True if goal was reached at any iteration |
| `success_postdisruption` | bool | True if goal reached after disruption (or no disruption) |
| `iterations` | int | Iteration count at termination |
| `path_length` | int | Number of moves in the found path |
| `optimal_length` | int | True optimal path length (A* reference) |
| `path_optimality` | float | path_length / optimal_length (lower is better) |
| `time_to_half_entropy` | float | Iterations from peak entropy to 50% of peak |
| `diversity_floor` | float | Minimum entropy reached during run |
| `mean_entropy` | float | Average Shannon entropy across all samples |
| `adaptation_time` | float | Iterations from disruption to 80% entropy recovery |
| `entropy_history` | str | JSON-encoded list of entropy samples (every 10 iters) |

## Maze Type Descriptions

- **Shortest Path Trap**: U-shaped dead end near the goal. Tests deception resistance.
- **Sudden Wall**: Short path blocked at iteration 100. Tests adaptation to non-stationarity.
- **Parallel Paths**: Two routes of equal length. Tests multimodal exploration.
