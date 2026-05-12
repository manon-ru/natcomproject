"""
ga_sensitivity_sweep.py — Sub-experiment investigating the GA encoding/parameter coupling.

The proposal-spec mutation_rate=0.3 was calibrated for Shrestha's 36-bit waypoint encoding.
Applied to our 160-gene direction encoding, expected mutations = 48 per offspring (destructive).
This script sweeps mutation_rate × chromosome_length to surface the relationship.

Usage:
    uv run python scripts/ga_sensitivity_sweep.py
    uv run python scripts/ga_sensitivity_sweep.py --trials 20    # more trials per cell
    uv run python scripts/ga_sensitivity_sweep.py --maze "Sudden Wall"  # different maze

Outputs:
    results/ga_sensitivity.csv   — one row per trial
    figures/ga_sensitivity.png   — success-rate heatmap
"""
import argparse
import csv
import json
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

from config import MAZE_WIDTH, MAZE_HEIGHT, DISRUPTION_TIME, ITERATION_LIMIT_MULTIPLIER, FORCED_MIN_ITERATIONS_AFTER_DISRUPTION
from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm
from runner import _seed_worker


def _run_one(task):
    """Worker: (mutation_rate, chromosome_length, maze_type, instance_seed, trial_seed) -> dict"""
    mutation_rate, chromosome_length, maze_type, instance_seed, trial_seed = task

    maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT, seed=instance_seed, maze_type=maze_type)
    initial_optimal_path = maze.shortest_path()
    if initial_optimal_path is None:
        return None
    optimal_steps = len(initial_optimal_path) - 1

    disruption_iteration = DISRUPTION_TIME if maze_type == "Sudden Wall" else -1
    forced_min = (DISRUPTION_TIME + FORCED_MIN_ITERATIONS_AFTER_DISRUPTION) if maze_type == "Sudden Wall" else 0

    if maze_type == "Sudden Wall" and getattr(maze, "dynamic_wall", None):
        w1, w2 = maze.dynamic_wall
        maze.add_wall(w1, w2)
        post_path = maze.shortest_path()
        maze.remove_wall(w1, w2)
        if post_path is not None:
            optimal_steps = len(post_path) - 1

    max_iterations = max(int(optimal_steps * ITERATION_LIMIT_MULTIPLIER), forced_min + 50)

    _seed_worker(trial_seed)
    ga = GeneticAlgorithm(
        maze,
        pop_size=50,
        chromosome_length=chromosome_length,
        crossover_rate=0.5,
        mutation_rate=mutation_rate,
    )
    result = ga.run(
        max_iterations=max_iterations,
        disruption_iteration=disruption_iteration,
        forced_min_iterations=forced_min,
    )

    path_length = len(result["path"]) - 1 if result.get("path") else None
    return {
        "mutation_rate": mutation_rate,
        "chromosome_length": chromosome_length,
        "maze_type": maze_type,
        "instance_seed": instance_seed,
        "trial_seed": trial_seed,
        "success": bool(result["success"]),
        "iterations": result["iterations"],
        "path_length": path_length,
        "optimal_length": optimal_steps,
        "path_optimality": path_length / optimal_steps if (path_length and optimal_steps > 0) else None,
        "mean_entropy": sum(ga.entropy_history) / len(ga.entropy_history) if ga.entropy_history else None,
    }


def main():
    ap = argparse.ArgumentParser(description="GA sensitivity sweep: mutation_rate × chromosome_length")
    ap.add_argument("--maze", default="U-Trap", choices=["U-Trap", "Sudden Wall", "Parallel Paths"])
    ap.add_argument("--trials", type=int, default=10, help="trials per cell (default 10)")
    ap.add_argument("--instances", type=int, default=1, help="distinct maze instances (default 1)")
    args = ap.parse_args()

    mutation_rates = [0.01, 0.05, 0.1, 0.3]
    chromosome_lengths = [80, 160]

    tasks = []
    for mr in mutation_rates:
        for cl in chromosome_lengths:
            for inst in range(1, args.instances + 1):
                for t in range(args.trials):
                    tasks.append((mr, cl, args.maze, inst, inst * 1000 + t))

    total = len(tasks)
    print(f"═══ GA Sensitivity Sweep ═══")
    print(f"  Maze:               {args.maze}")
    print(f"  Mutation rates:     {mutation_rates}")
    print(f"  Chromosome lengths: {chromosome_lengths}")
    print(f"  Trials per cell:    {args.trials}")
    print(f"  Instances per cell: {args.instances}")
    print(f"  Total runs:         {total}")
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
            if i % 20 == 0 or i == total:
                print(f"  [{i}/{total}] {time.time() - start:.1f}s")

    print(f"\nCompleted {len(results)} runs in {time.time() - start:.1f}s")
    print()

    # ── CSV output ────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ga_sensitivity.csv"
    header = ["mutation_rate", "chromosome_length", "maze_type", "instance_seed", "trial_seed",
              "success", "iterations", "path_length", "optimal_length", "path_optimality", "mean_entropy"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in results:
            w.writerow({k: (r[k] if r[k] is not None else "") for k in header})
    print(f"CSV: {csv_path} ({len(results)} rows)")

    # ── Aggregate table ───────────────────────────────────────────────────────
    print()
    print(f"═══ Success Rate ({args.maze}) ═══")
    print(f"  {'mut\\CL':<8}", *[f"{cl:>6d}" for cl in chromosome_lengths])
    for mr in mutation_rates:
        row = [f"  {mr:<8.2f}"]
        for cl in chromosome_lengths:
            cell = [r for r in results if r["mutation_rate"] == mr and r["chromosome_length"] == cl]
            if cell:
                sr = sum(1 for r in cell if r["success"]) / len(cell)
                row.append(f"{sr:>6.0%}")
            else:
                row.append(f"{'N/A':>6s}")
        print(" ".join(row))

    # ── Heatmap figure ────────────────────────────────────────────────────────
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    heat = np.zeros((len(mutation_rates), len(chromosome_lengths)))
    for i, mr in enumerate(mutation_rates):
        for j, cl in enumerate(chromosome_lengths):
            cell = [r for r in results if r["mutation_rate"] == mr and r["chromosome_length"] == cl]
            heat[i, j] = sum(1 for r in cell if r["success"]) / len(cell) if cell else 0.0

    im = ax.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(chromosome_lengths)))
    ax.set_xticklabels([f"L={cl}" for cl in chromosome_lengths])
    ax.set_yticks(range(len(mutation_rates)))
    ax.set_yticklabels([f"μ={mr}" for mr in mutation_rates])
    ax.set_xlabel("Chromosome length")
    ax.set_ylabel("Mutation rate")
    ax.set_title(f"GA success rate — {args.maze} 40×40\n"
                 f"(proposal-spec params are μ=0.3, L=160 → top-right)")

    for i in range(len(mutation_rates)):
        for j in range(len(chromosome_lengths)):
            ax.text(j, i, f"{heat[i, j]:.0%}", ha="center", va="center",
                    color="white" if heat[i, j] < 0.5 else "black", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Success rate")
    # Annotate the proposal-spec cell
    ax.add_patch(plt.Rectangle((len(chromosome_lengths) - 1.5, len(mutation_rates) - 1.5), 1, 1,
                               fill=False, edgecolor="blue", lw=3))
    fig.tight_layout()
    fig_path = "figures/ga_sensitivity.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"\nFigure: {fig_path}")
    print()
    print("→ Blue box highlights the proposal-spec configuration (μ=0.3, L=160).")
    print("→ Compare against lower mutation rates and shorter chromosomes to see the encoding sensitivity.")


if __name__ == "__main__":
    main()
