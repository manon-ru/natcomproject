# Task 22: Pilot Validation

Source: `results/runs_pilot.csv` (27 rows from `uv run python main.py --pilot`)

## Criterion 1: Coverage

Pilot produced 27/27 expected rows (3 algorithms × 3 maze types × 3 population sizes).

Result: **PASS**

## Criterion 2: Non-empty entropy_history

Every row has a non-empty `entropy_history` JSON list.

Result: **PASS**

## Criterion 3: At least one success per (algorithm, maze) pair

Success counts per (algorithm, maze) pair:

| Algorithm | Maze type           | Successes / 3 pop sizes |
|-----------|---------------------|-------------------------|
| ACO       | Parallel Paths      | 3/3                     |
| ACO       | Shortest Path Trap  | 3/3                     |
| ACO       | Sudden Wall         | 3/3                     |
| GA        | Parallel Paths      | 3/3                     |
| GA        | Shortest Path Trap  | 0/3                     |
| GA        | Sudden Wall         | 0/3                     |
| PSO       | Parallel Paths      | 0/3                     |
| PSO       | Shortest Path Trap  | 0/3                     |
| PSO       | Sudden Wall         | 0/3                     |

Result: **SOFT-FAIL** (GA + PSO show 0% on harder mazes)

## Interpretation

The Criterion 3 soft-fail surfaces a documented research finding:

- **GA on Shortest Path Trap / Sudden Wall**: consistent with `runs.csv.pre-fix`
  data — GA was already failing on these mazes in the pre-canonical baseline.
  Not a regression introduced by the PSO/ACO refactor.

- **PSO on all mazes**: this is the anticipated parameter-choice finding the
  plan flagged before execution. With ω = 1.0, c1 = 0.1, c2 = 0.2 drawn from
  Shrestha et al. for a waypoint-based formulation, the canonical
  continuous-PSO + grid discretization rarely converges to the goal cell in
  the iteration budget. The plan's pre-pilot oracle review predicted this
  ("PSO with Shrestha params on grid is mathematically dubious for
  convergence; document as research finding"). The methodology section of
  the report should record this as evidence about parameter portability
  across PSO formulations.

## Decision

Per plan Wave 5 instructions (option (a)): proceed with the full 2700-run
experiment and report the PSO under-convergence as a research finding rather
than blocking the experiment on parameter retuning.

Both hard criteria (1 and 2) PASS. Pilot gate cleared.
