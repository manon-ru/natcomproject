"""
hypothesis_report.py - Evaluate proposal hypotheses H1/H2/H3 against a runs.csv.

Produces a structured text report that maps each subordinate proposal
prediction to its observed direction in the data. Predictions are stated in
proposal.txt sections "Hypotheses" (H1, H2, H3); the mapping to columns is:

    Time to 50% entropy loss   -> time_to_half_entropy   (lower = faster collapse)
    Diversity floor            -> diversity_floor        (lower = more severe collapse)
    Mean entropy               -> mean_entropy           (higher = sustained diversity)
    Adaptation time (Maze 2)   -> adaptation_time        (lower = faster recovery)
    Success rate               -> success_overall
    Path optimality            -> path_optimality        (1.0 = matches A* optimum)

For each prediction we print: predicted ordering, observed ordering, and a
single-line verdict (supported / partial / contradicted). Numerical aggregation
uses per-cell means with NaN-skipping. We do not run hypothesis tests; the
report is descriptive, intended as input to the manual write-up.

Usage:
    uv run python scripts/hypothesis_report.py --input results/runs.csv \
        --output results/hypothesis_report.txt
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict

ALGOS = ("GA", "PSO", "ACO")
MAZES = ("Shortest Path Trap", "Sudden Wall", "Parallel Paths")
POPS = (20, 50, 150)


def _f(s):
    if s in ("", None):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _mean_skip_nan(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.fmean(xs) if xs else float("nan")


def _load(path):
    by_cell = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["algo"], row["maze_type"], int(row["pop_size"]))
            by_cell[key].append({
                "success": row["success_overall"].lower() == "true",
                "success_post": row["success_postdisruption"].lower() == "true",
                "iters": _f(row["iterations"]),
                "optimality": _f(row["path_optimality"]),
                "t_half": _f(row["time_to_half_entropy"]),
                "floor": _f(row["diversity_floor"]),
                "mean_ent": _f(row["mean_entropy"]),
                "adapt": _f(row["adaptation_time"]),
            })
    return by_cell


def _cell_stat(by_cell, key, field):
    rows = by_cell.get(key, [])
    if not rows:
        return float("nan")
    return _mean_skip_nan([r[field] for r in rows])


def _verdict(observed_rank, predicted_rank):
    if not all(a == p for a, p in zip(observed_rank, predicted_rank)):
        ordered = observed_rank[0] == predicted_rank[0]
        return "PARTIAL (worst matches)" if ordered else "CONTRADICTED"
    return "SUPPORTED"


def _rank(values, ascending=True):
    items = sorted(values.items(), key=lambda kv: (math.isnan(kv[1]), kv[1] if ascending else -kv[1]))
    return [k for k, _ in items]


def _line(parts, widths):
    return "  ".join(p.ljust(w) for p, w in zip(parts, widths))


def report_h1(by_cell, pop):
    lines = [f"--- H1 (Coupling shapes diversity dynamics)  [pop={pop}] ---"]
    lines.append("Prediction: PSO collapses fastest, GA most stable, ACO in between.")
    lines.append("")
    metrics = [
        ("time_to_half_entropy (lower = faster collapse)", "t_half", True,  ("PSO", "ACO", "GA")),
        ("diversity_floor      (lower = severer collapse)", "floor",  True,  ("PSO", "ACO", "GA")),
        ("mean_entropy         (higher = sustained diversity)", "mean_ent", False, ("GA", "ACO", "PSO")),
    ]
    for maze in MAZES:
        lines.append(f"  Maze: {maze}")
        for label, field, asc, pred in metrics:
            vals = {algo: _cell_stat(by_cell, (algo, maze, pop), field) for algo in ALGOS}
            obs = _rank(vals, ascending=asc)
            sep = " < " if asc else " > "
            v_str = "  ".join(f"{a}={vals[a]:.3f}" for a in ALGOS if not math.isnan(vals[a]))
            lines.append(f"    {label}")
            lines.append(f"      predicted: {sep.join(pred)}")
            lines.append(f"      observed:  {sep.join(obs)}")
            lines.append(f"      values:    {v_str}")
            lines.append(f"      verdict:   {_verdict(obs, pred)}")
        lines.append("")
    return lines


def report_h2(by_cell, pop):
    lines = [f"--- H2 (Environmental challenge x coupling)  [pop={pop}] ---"]
    sub_h2 = [
        ("Shortest Path Trap (deception)  most stresses PSO",
         "Shortest Path Trap", "floor", True,  "PSO"),
        ("Sudden Wall (non-stat)    most stresses ACO (adaptation time)",
         "Sudden Wall",   "adapt", False, "ACO"),
        ("Parallel Paths (multimodality)  most stresses PSO (diversity floor)",
         "Parallel Paths", "floor", True,  "PSO"),
    ]
    for label, maze, field, asc_means_worst, predicted_worst in sub_h2:
        vals = {algo: _cell_stat(by_cell, (algo, maze, pop), field) for algo in ALGOS}
        ranked = _rank(vals, ascending=asc_means_worst)
        observed_worst = ranked[0]
        v_str = "  ".join(f"{a}={vals[a]:.3f}" for a in ALGOS if not math.isnan(vals[a]))
        verdict = "SUPPORTED" if observed_worst == predicted_worst else f"CONTRADICTED (worst={observed_worst})"
        lines.append(f"  {label}")
        lines.append(f"    field={field}  predicted-worst={predicted_worst}  observed-worst={observed_worst}")
        lines.append(f"    values: {v_str}")
        lines.append(f"    verdict: {verdict}")
        lines.append("")
    return lines


def report_h3(by_cell):
    lines = ["--- H3 (Population scale modulates but does not reverse) ---"]
    lines.append("Prediction: larger pop amplifies PSO weakness; GA benefits; ACO insensitive.")
    lines.append("")
    for algo in ALGOS:
        for maze in MAZES:
            vals = {pop: _cell_stat(by_cell, (algo, maze, pop), "floor") for pop in POPS}
            v_str = "  ".join(f"pop{p}={vals[p]:.3f}" for p in POPS if not math.isnan(vals[p]))
            # Trend: monotone-down = collapses more with scale; monotone-up = recovers with scale
            trend = "?"
            if all(not math.isnan(vals[p]) for p in POPS):
                if vals[20] > vals[50] > vals[150]:
                    trend = "DECREASING (collapse deepens with scale)"
                elif vals[20] < vals[50] < vals[150]:
                    trend = "INCREASING (scale helps)"
                else:
                    trend = "NON-MONOTONIC"
            lines.append(f"  {algo:4s} {maze:15s}  diversity_floor:  {v_str}   trend: {trend}")
    lines.append("")
    return lines


def report_success_optimality(by_cell):
    lines = ["--- Context: success rate and path optimality ---"]
    lines.append(f"{'Algo':6}{'Maze':18}{'Pop':5}{'Success':>10}{'Optimality':>12}")
    for algo in ALGOS:
        for maze in MAZES:
            for pop in POPS:
                rows = by_cell.get((algo, maze, pop), [])
                if not rows:
                    continue
                n = len(rows)
                s = sum(1 for r in rows if r["success"])
                opt = _mean_skip_nan([r["optimality"] for r in rows if r["success"]])
                lines.append(f"{algo:6}{maze:18}{pop:<5}{f'{s}/{n} ({100*s//n if n else 0}%)':>10}{opt:>12.3f}")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/runs.csv")
    ap.add_argument("--output", default="results/hypothesis_report.txt")
    args = ap.parse_args()

    by_cell = _load(args.input)
    total = sum(len(v) for v in by_cell.values())

    out = []
    out.append(f"Hypothesis report from {args.input}")
    out.append(f"Total rows: {total}  Cells: {len(by_cell)}/27")
    out.append("")
    out.extend(report_success_optimality(by_cell))
    for pop in POPS:
        out.extend(report_h1(by_cell, pop))
    for pop in POPS:
        out.extend(report_h2(by_cell, pop))
    out.extend(report_h3(by_cell))

    text = "\n".join(out) + "\n"
    with open(args.output, "w") as f:
        f.write(text)
    print(f"Wrote {args.output}  ({total} rows analyzed)")


if __name__ == "__main__":
    main()
