2026-05-12: time_to_half_entropy must measure from peak_index and scale by sample_interval.
[2026-05-12] adaptation_time floor behavior
- Used an explicit entropy_floor parameter rather than config import to keep the metric self-contained.
- Kept threshold_ratio=0.8 and sample_interval=10 unchanged.
4. [2026-05-12] ES baseline removal
- Removed OnePlusOneES from the package exports and main runner.
- Verified with a direct __all__ check and grep-based reference count.
## [2026-05-12] uv env required
- Use `uv run python ...` not bare `python` for all scripts in this project. `.venv` has numpy/matplotlib.
- T7: ALL_REACHABLE 30/30 confirmed. Sudden Wall pre=39, post=49-57 (wing detour post-disruption).
 - Task 14: `src/visualization/demo.py` remains a scratch-only preview; keep proposal figures in `scripts/figures.py` and avoid ES-era labels like "Baseline GA".

## Task 9: ACO Parameter Alignment

- ACO constructor now: `beta=5.0`, `pheromone_deposit=2.0`, `initial_pheromone=0.8`
- `self.initial_pheromone` stored on instance for reference
- `self.pheromones` initialized via `* initial_pheromone` (not hardcoded 0.1)
- Module docstring added documenting per-cell vs per-edge pheromone deviation from Akka & Khaber
- QA command confirms all 5 assertions pass: alpha=1.0, beta=5.0, Q=2.0, rho=0.1, tau0=0.8
- Evidence saved to `.sisyphus/evidence/task-9-aco-params.txt` (gitignored path)

## Task 8 – PSO omega fix
- `inertia = random.random()` was effectively randomizing the constant ω; replaced with `self.omega`
- `PSO.__init__` now: `omega: float = 1.0` between `num_particles` and `c1`
- The previous session had bundled PSO+ACO in one commit; needed soft-reset + two separate commits
- Evidence saved to `.sisyphus/evidence/task-8-pso-omega.txt` (gitignored)
