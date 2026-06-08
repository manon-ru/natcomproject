"""
Generate figures from the experiment results CSV.

Usage:
    uv run python scripts/figures.py --input results/runs.csv --output figures/
    uv run python scripts/figures.py --input results/runs_quick.csv --output figures_quick/
"""
import argparse
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Larger fonts so axes/labels stay readable when figures are shrunk in the report.
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import MAZE_TYPES, POPULATION_SIZES, ENTROPY_SAMPLE_INTERVAL

ALGORITHMS = ["GA", "PSO", "ACO"]
ALGO_COLORS = {"GA": "#3a86ff", "PSO": "#e07b39", "ACO": "#57cc99"}
MAZE_SLUGS = {
    "Shortest Path Trap": "shortest_path_trap",
    "Sudden Wall": "sudden_wall",
    "Parallel Paths": "parallel_paths",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_float(v):
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _parse_int(v):
    if v == "" or v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    if v in ("True", "true", "1"):
        return True
    return False


def load_results(path: str) -> list:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "algo": row["algo"],
                "maze_type": row["maze_type"],
                "pop_size": _parse_int(row["pop_size"]),
                "instance_seed": _parse_int(row["instance_seed"]),
                "trial_seed": _parse_int(row["trial_seed"]),
                "success_overall": _parse_bool(row["success_overall"]),
                "success_postdisruption": _parse_bool(row["success_postdisruption"]),
                "iterations": _parse_int(row["iterations"]),
                "path_length": _parse_int(row["path_length"]),
                "optimal_length": _parse_int(row["optimal_length"]),
                "path_optimality": _parse_float(row["path_optimality"]),
                "time_to_half_entropy": _parse_float(row["time_to_half_entropy"]),
                "diversity_floor": _parse_float(row["diversity_floor"]),
                "mean_entropy": _parse_float(row["mean_entropy"]),
                "adaptation_time": _parse_float(row["adaptation_time"]),
                "entropy_history": json.loads(row["entropy_history"]) if row["entropy_history"] else [],
            })
    return rows


def _filter(rows, **kwargs):
    result = rows
    for k, v in kwargs.items():
        result = [r for r in result if r[k] == v]
    return result


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sem(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    variance = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance / len(vals))


# ── Figure 1: Entropy curves ─────────────────────────────────────────────────

def figure_entropy_curves(rows: list, out_dir: str) -> None:
    """One figure per (maze_type, pop_size) — 9 total. Each shows 3 algorithm lines."""
    pop_sizes = sorted(set(r["pop_size"] for r in rows if r["pop_size"] is not None))
    maze_types = [m for m in MAZE_TYPES if any(r["maze_type"] == m for r in rows)]

    for maze in maze_types:
        for pop in pop_sizes:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            has_data = False

            for algo in ALGORITHMS:
                cell = _filter(rows, maze_type=maze, pop_size=pop, algo=algo)
                if not cell:
                    continue
                histories = [r["entropy_history"] for r in cell if r["entropy_history"]]
                if not histories:
                    continue

                max_len = max(len(h) for h in histories)
                # Pad shorter histories with their last value
                padded = [h + [h[-1]] * (max_len - len(h)) if h else [] for h in histories]
                padded = [h for h in padded if h]
                if not padded:
                    continue

                arr = np.array(padded, dtype=float)
                mean_curve = arr.mean(axis=0)
                sem_curve = arr.std(axis=0, ddof=1) / np.sqrt(len(padded)) if len(padded) > 1 else np.zeros(len(mean_curve))
                iters = [i * ENTROPY_SAMPLE_INTERVAL for i in range(len(mean_curve))]

                color = ALGO_COLORS.get(algo, "gray")
                ax.plot(iters, mean_curve, label=algo, color=color, lw=2)
                ax.fill_between(iters, mean_curve - sem_curve, mean_curve + sem_curve,
                                alpha=0.2, color=color)
                has_data = True

            if maze == "Sudden Wall":
                from config import DISRUPTION_TIME
                ax.axvline(x=DISRUPTION_TIME, color="black", linestyle="--", lw=1, alpha=0.6, label=f"Disruption T={DISRUPTION_TIME}")

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Shannon Entropy (bits)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

            slug = MAZE_SLUGS.get(maze, maze.lower().replace(" ", "_"))
            fname = os.path.join(out_dir, f"entropy_{slug}_pop{pop}.png")
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close(fig)
            if has_data:
                print(f"  Saved {fname}")


# ── Figure 2: Success rate ────────────────────────────────────────────────────

def figure_success_rate(rows: list, out_dir: str) -> None:
    """One figure per maze type. Grouped bars: x=pop_size, groups=algorithm."""
    maze_types = [m for m in MAZE_TYPES if any(r["maze_type"] == m for r in rows)]
    pop_sizes = sorted(set(r["pop_size"] for r in rows if r["pop_size"] is not None))

    for maze in maze_types:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        x = np.arange(len(pop_sizes))
        width = 0.25
        offsets = [-width, 0, width]

        for i, algo in enumerate(ALGORITHMS):
            rates = []
            for pop in pop_sizes:
                cell = _filter(rows, maze_type=maze, pop_size=pop, algo=algo)
                if cell:
                    rates.append(sum(1 for r in cell if r["success_overall"]) / len(cell))
                else:
                    rates.append(0.0)
            ax.bar(x + offsets[i], rates, width, label=algo,
                   color=ALGO_COLORS.get(algo, "gray"), alpha=0.85)

        ax.set_xlabel("Population Size")
        ax.set_ylabel("Success Rate")
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in pop_sizes])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        slug = MAZE_SLUGS.get(maze, maze.lower().replace(" ", "_"))
        fname = os.path.join(out_dir, f"success_rate_{slug}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Figure 3: Adaptation time (Sudden Wall only) ──────────────────────────────

def figure_adaptation_time(rows: list, out_dir: str) -> None:
    """Bar chart: x=algorithm, bars=pop_size. Only for Sudden Wall."""
    sw_rows = _filter(rows, maze_type="Sudden Wall")
    if not sw_rows:
        return

    pop_sizes = sorted(set(r["pop_size"] for r in sw_rows if r["pop_size"] is not None))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ALGORITHMS))
    width = 0.25
    n_pops = len(pop_sizes)
    offsets = [width * (i - (n_pops - 1) / 2) for i in range(n_pops)]

    for i, pop in enumerate(pop_sizes):
        means = []
        for algo in ALGORITHMS:
            cell = _filter(sw_rows, pop_size=pop, algo=algo)
            vals = [r["adaptation_time"] for r in cell if r["adaptation_time"] is not None]
            means.append(_mean(vals))
        bar_vals = [v if v is not None else 0 for v in means]
        ax.bar(x + offsets[i], bar_vals, width, label=f"pop={pop}", alpha=0.85)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Adaptation Time (iterations)")
    ax.set_xticks(x)
    ax.set_xticklabels(ALGORITHMS)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fname = os.path.join(out_dir, "adaptation_time_sudden_wall.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Figure 4: Path optimality box-plots ──────────────────────────────────────

def figure_path_optimality(rows: list, out_dir: str) -> None:
    """Box plot per maze type. x=algorithm, boxes=pop_size. Successful runs only."""
    maze_types = [m for m in MAZE_TYPES if any(r["maze_type"] == m for r in rows)]
    pop_sizes = sorted(set(r["pop_size"] for r in rows if r["pop_size"] is not None))

    for maze in maze_types:
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = []
        data = []
        labels = []
        colors = []
        x_ticks = []
        x_labels = []

        group_width = len(pop_sizes) + 1
        for gi, algo in enumerate(ALGORITHMS):
            x_center = gi * group_width + (len(pop_sizes) - 1) / 2
            x_ticks.append(x_center)
            x_labels.append(algo)
            for pi, pop in enumerate(pop_sizes):
                cell = _filter(rows, maze_type=maze, pop_size=pop, algo=algo)
                vals = [r["path_optimality"] for r in cell
                        if r["path_optimality"] is not None and r["success_overall"]]
                pos = gi * group_width + pi
                positions.append(pos)
                data.append(vals if vals else [0])
                labels.append(f"pop={pop}")
                colors.append(ALGO_COLORS.get(algo, "gray"))

        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                        medianprops={"color": "black", "lw": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(y=1.0, color="red", linestyle="--", lw=1, alpha=0.7, label="Optimal (ratio=1.0)")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Path Length / Optimal Length")
        ax.set_title(f"Path Optimality — {maze}")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        slug = MAZE_SLUGS.get(maze, maze.lower().replace(" ", "_"))
        fname = os.path.join(out_dir, f"path_optimality_{slug}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=100)
        plt.close(fig)
        print(f"  Saved {fname}")


# ── LaTeX summary table ───────────────────────────────────────────────────────

def figure_latex_summary_table(rows: list, out_dir: str) -> None:
    """27-row LaTeX table: one row per (algo, maze, pop) cell."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lllrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Algo & Maze & Pop & SR & Iters & Opt & $t_{50\%}$ & Floor & $\bar{H}$ & Adapt \\")
    lines.append(r"\midrule")

    for algo in ALGORITHMS:
        for maze in MAZE_TYPES:
            for pop in sorted(set(r["pop_size"] for r in rows if r["pop_size"] is not None)):
                cell = _filter(rows, algo=algo, maze_type=maze, pop_size=pop)
                if not cell:
                    continue

                sr = sum(1 for r in cell if r["success_overall"]) / len(cell)
                iters_vals = [r["iterations"] for r in cell if r["iterations"] is not None]
                opt_vals = [r["path_optimality"] for r in cell if r["path_optimality"] is not None and r["success_overall"]]
                tth_vals = [r["time_to_half_entropy"] for r in cell if r["time_to_half_entropy"] is not None]
                floor_vals = [r["diversity_floor"] for r in cell if r["diversity_floor"] is not None]
                ment_vals = [r["mean_entropy"] for r in cell if r["mean_entropy"] is not None]
                adapt_vals = [r["adaptation_time"] for r in cell if r["adaptation_time"] is not None]

                def fmt(v, decimals=2):
                    if v is None:
                        return "N/A"
                    return f"{v:.{decimals}f}"

                maze_short = {"Shortest Path Trap": "SPT", "Sudden Wall": "SW", "Parallel Paths": "PP"}.get(maze, maze)
                row_str = (
                    f"{algo} & {maze_short} & {pop} & "
                    f"{sr:.2f} & "
                    f"{fmt(_mean(iters_vals), 0)} & "
                    f"{fmt(_mean(opt_vals))} & "
                    f"{fmt(_mean(tth_vals), 0)} & "
                    f"{fmt(_mean(floor_vals))} & "
                    f"{fmt(_mean(ment_vals))} & "
                    f"{fmt(_mean(adapt_vals), 0)} \\\\"
                )
                lines.append(row_str)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Summary metrics per experimental cell (3 algorithms × 3 maze types × 3 population sizes). SR=success rate, Iters=mean iterations, Opt=mean path optimality, $t_{50\%}$=time to 50\% entropy loss, Floor=diversity floor, $\bar{H}$=mean entropy, Adapt=adaptation time (Sudden Wall only).}")
    lines.append(r"\label{tab:summary}")
    lines.append(r"\end{table}")

    fname = os.path.join(out_dir, "summary_table.tex")
    with open(fname, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate report figures from experiment CSV.")
    ap.add_argument("--input", required=True, help="Path to results CSV")
    ap.add_argument("--output", required=True, help="Output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading {args.input}...")
    rows = load_results(args.input)
    print(f"Loaded {len(rows)} rows.")

    if not rows:
        print("WARNING: No data rows found. Figures will be empty.", file=sys.stderr)
        sys.exit(1)

    print("Generating entropy curves...")
    figure_entropy_curves(rows, args.output)

    print("Generating success rate bars...")
    figure_success_rate(rows, args.output)

    print("Generating adaptation time bars...")
    figure_adaptation_time(rows, args.output)

    print("Generating path optimality box-plots...")
    figure_path_optimality(rows, args.output)

    print("Generating LaTeX summary table...")
    figure_latex_summary_table(rows, args.output)

    print(f"\nDone. Figures in {args.output}")


if __name__ == "__main__":
    main()
