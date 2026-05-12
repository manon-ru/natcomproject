2026-05-12: Direct import of evaluation.metrics in the default environment failed because numpy is not installed; used isolated module loading with stubbed maze dependency for QA.

## Task 9: Incidental pso.py in commit

`src/algorithms/pso.py` (omega param addition from a prior task) was already staged when task-9 began, so it got bundled into the `fix(aco)` commit. The ACO file itself is the only file intentionally modified by task-9. The pso.py changes were a prior task's staged work and appear correct (omega inertia weight).

## Task 14: Demo import diagnostics

Pyright still reports missing numpy/matplotlib imports in the IDE, but `uv run python -c "import sys; sys.path.insert(0,'src'); import visualization.demo; print('OK')"` succeeds in the project env.
