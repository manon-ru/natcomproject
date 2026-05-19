"""
aggregate_summary.py - Build results/aggregate_summary.txt from a runs.csv.

Reads the experiment CSV, groups by (algo, maze_type, pop_size), and writes a
fixed-width table with success rates. Output format matches the existing
results/aggregate_summary.txt so downstream consumers (report tables) remain
stable across re-runs.

Usage:
    uv run python scripts/aggregate_summary.py \
        --input results/runs.csv \
        --output results/aggregate_summary.txt
"""
import argparse
import csv
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/runs.csv")
    ap.add_argument("--output", default="results/aggregate_summary.txt")
    args = ap.parse_args()

    successes = defaultdict(int)
    totals = defaultdict(int)
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["algo"], row["maze_type"], int(row["pop_size"]))
            totals[key] += 1
            if row["success_overall"].lower() == "true":
                successes[key] += 1

    keys = sorted(totals.keys(), key=lambda k: (k[0], k[1], k[2]))

    lines = ["Algo   Maze               Pop   Success Rate  "]
    for algo, maze, pop in keys:
        s = successes[(algo, maze, pop)]
        t = totals[(algo, maze, pop)]
        rate_pct = round(100.0 * s / t) if t else 0
        lines.append(f"{algo:<7}{maze:<19}{pop:<6}{rate_pct}% ({s}/{t})")

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output} ({len(keys)} cells, {sum(totals.values())} total runs)")


if __name__ == "__main__":
    main()
