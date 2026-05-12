"""
ga_iteration_budget_sweep.py — Investigate whether GA can succeed given more iterations.

Tests proposal-spec GA (mutation_rate=0.3, chromosome_length=160, pop=50) on U-Trap 40×40
across iteration budgets ranging from 610 (proposal) to 50,000.

Hypothesis: if GA's failure is iteration-bound, larger budgets should yield non-zero
success rates. If GA still fails at 50k generations, the encoding (not the budget)
is the binding constraint.

Usage:
    uv run python scripts/ga_iteration_budget_sweep.py
    uv run python scripts/ga_iteration_budget_sweep.py --trials 6   # more trials
"""
import argparse
import csv
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import MAZE_WIDTH, MAZE_HEIGHT
from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm
from runner import _seed_worker


def _run_one(task):
    """(max_iterations, instance_seed, trial_seed) -> dict"""
    max_iters, instance_seed, trial_seed = task

    maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT, seed=instance_seed, maze_type="U-Trap")
    initial_optimal = maze.shortest_path()
    if initial_optimal is None:
        return None
    optimal_steps = len(initial_optimal) - 1

    _seed_worker(trial_seed)
    ga = GeneticAlgorithm(
        maze,
        pop_size=50,
        chromosome_length=160,   # proposal-spec
        crossover_rate=0.5,
        mutation_rate=0.3,        # proposal-spec
    )

    t0 = time.time()
    result = ga.run(max_iterations=max_iters, disruption_iteration=-1, forced_min_iterations=0)
    wall = time.time() - t0

    # Compute best-found Manhattan distance to goal across the final population
    gx, gy = maze.goal
    end_pos = result["path"][-1] if result.get("path") else maze.start
    best_dist = abs(end_pos[0] - gx) + abs(end_pos[1] - gy)

    path_length = len(result["path"]) - 1 if result.get("path") else 0

    return {
        "max_iterations": max_iters,
        "instance_seed": instance_seed,
        "trial_seed": trial_seed,
        "success": bool(result["success"]),
        "iterations": result["iterations"],
        "path_length": path_length,
        "optimal_length": optimal_steps,
        "best_distance_to_goal": best_dist,
        "wall_time_sec": round(wall, 2),
        "mean_entropy": (sum(ga.entropy_history) / len(ga.entropy_history)) if ga.entropy_history else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4, help="trials per budget (default 4)")
    args = ap.parse_args()

    budgets = [610, 5000, 15000, 50000]
    instance_seed = 1

    tasks = []
    for b in budgets:
        for t in range(args.trials):
            tasks.append((b, instance_seed, instance_seed * 1000 + t))

    total = len(tasks)
    print(f"═══ GA Iteration Budget Sweep ═══")
    print(f"  Params: pop=50, μ=0.3, L=160 (proposal-spec)")
    print(f"  Maze: U-Trap 40×40, seed={instance_seed}")
    print(f"  Budgets: {budgets}")
    print(f"  Trials per budget: {args.trials}")
    print(f"  Total runs: {total}")
    print()

    multiprocessing.set_start_method("spawn", force=True)
    num_workers = min(os.cpu_count() or 4, 8)
    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_run_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is not None:
                results.append(r)
                print(f"  [{i}/{total}] budget={r['max_iterations']} success={r['success']} "
                      f"dist={r['best_distance_to_goal']} wall={r['wall_time_sec']:.1f}s")

    print(f"\nCompleted {len(results)} runs in {time.time() - start:.1f}s")
    print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ga_iteration_budget.csv"
    header = ["max_iterations", "instance_seed", "trial_seed", "success", "iterations",
              "path_length", "optimal_length", "best_distance_to_goal", "wall_time_sec", "mean_entropy"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in results:
            w.writerow({k: (r[k] if r[k] is not None else "") for k in header})
    print(f"CSV: {csv_path} ({len(results)} rows)")

    # ── Aggregate table ───────────────────────────────────────────────────────
    print()
    print(f"═══ Summary ═══")
    print(f"  {'Budget':<10} {'SR':<8} {'mean_dist_to_goal':<20} {'mean_wall_sec':<14}")
    summary = []
    for b in budgets:
        cell = [r for r in results if r["max_iterations"] == b]
        if not cell:
            continue
        sr = sum(1 for r in cell if r["success"]) / len(cell)
        mean_dist = sum(r["best_distance_to_goal"] for r in cell) / len(cell)
        mean_wall = sum(r["wall_time_sec"] for r in cell) / len(cell)
        print(f"  {b:<10} {sr:<8.0%} {mean_dist:<20.1f} {mean_wall:<14.1f}")
        summary.append((b, sr, mean_dist, mean_wall))

    # ── Figure: dual-axis line plot ───────────────────────────────────────────
    os.makedirs("figures", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))

    budgets_x = [s[0] for s in summary]
    srs = [s[1] for s in summary]
    dists = [s[2] for s in summary]

    color1 = "tab:blue"
    ax1.set_xlabel("Max iterations (log scale)")
    ax1.set_ylabel("Success rate", color=color1)
    ax1.plot(budgets_x, srs, "o-", color=color1, lw=2, markersize=10, label="Success rate")
    ax1.set_xscale("log")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Mean best distance to goal (cells)", color=color2)
    ax2.plot(budgets_x, dists, "s--", color=color2, lw=2, markersize=8, label="Mean dist to goal")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.invert_yaxis()   # smaller = better

    # Reference lines
    ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(x=610, color="gray", linestyle=":", alpha=0.4)
    ax1.text(610, 0.5, "proposal\nbudget", ha="center", va="center", fontsize=8, alpha=0.6)

    plt.title(f"GA on U-Trap 40×40 (proposal-spec: pop=50, μ=0.3, L=160)\n"
              f"Success rate and best-found distance vs iteration budget")
    fig.tight_layout()

    fig_path = "figures/ga_iteration_budget.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {fig_path}")

    # ── Interpretation hint ───────────────────────────────────────────────────
    print()
    print("Interpretation:")
    all_zero = all(s[1] == 0.0 for s in summary)
    if all_zero:
        print("  ✗ All budgets: 0% success — encoding is the binding constraint,")
        print("    not iteration count. Even 50,000 generations × 50 pop = 2.5M evaluations")
        print("    is insufficient when starting from random direction strings.")
    else:
        first_success = next((b for b, sr, _, _ in summary if sr > 0), None)
        print(f"  ✓ GA can succeed given enough budget. First non-zero success at {first_success} iterations.")
        print(f"    This rules out 'encoding is fundamentally inadequate' — instead, GA needs ~{first_success/610:.0f}x")
        print(f"    more compute than the proposal allows to converge under direction-string encoding.")


if __name__ == "__main__":
    main()
