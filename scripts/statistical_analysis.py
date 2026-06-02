"""
Statistical analysis of the experiment results, following the plan in the report.

It produces three things from a runs.csv:
- bootstrap 95% confidence intervals for each metric in every cell,
- pairwise Mann-Whitney U tests between algorithms within each maze and population
  cell, with Holm correction and rank-biserial effect sizes,
- factorial tests for the population main effect and the algorithm-by-population
  interaction.

Usage:
    uv run python scripts/statistical_analysis.py \
        --input results/runs.csv --outdir results
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ALGOS = ["GA", "PSO", "ACO"]
PAIRS = [("GA", "PSO"), ("GA", "ACO"), ("PSO", "ACO")]
N_BOOT = 10_000
SEED = 0

# Metrics recorded for every run.
ALL_RUN_METRICS = ["success_overall", "iterations", "diversity_floor", "mean_entropy"]
# Valid only for successful runs (path ratio is a partial path on failures).
SUCCESS_ONLY = ["path_optimality"]
# Recorded only when entropy actually halves, so often missing.
SPARSE = ["time_to_half_entropy"]
PAIRWISE_METRICS = ALL_RUN_METRICS + SUCCESS_ONLY + SPARSE
MIN_GROUP = 5


def metric_values(df, metric):
    """Per-run values for a metric, dropping rows where it is not defined."""
    if metric == "success_overall":
        return df["success_overall"].astype(int).to_numpy()
    s = df[metric]
    if metric in SUCCESS_ONLY:
        s = s[df["success_overall"]]
    return s.dropna().to_numpy(dtype=float)


def bootstrap_ci(x, rng, n_boot=N_BOOT, ci=95):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan, np.nan, 0
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(x.mean()), float(lo), float(hi), int(x.size)


def rank_biserial(u1, n1, n2):
    """Effect size from the U of the first group; >0 means it tends to be larger."""
    return 2.0 * u1 / (n1 * n2) - 1.0


def mann_whitney(x, y):
    if x.size < MIN_GROUP or y.size < MIN_GROUP:
        return None
    if np.ptp(np.concatenate([x, y])) == 0:
        return dict(u=x.size * y.size / 2, p=1.0, r=0.0, n1=x.size, n2=y.size)
    u1, p = mannwhitneyu(x, y, alternative="two-sided")
    return dict(u=u1, p=p, r=rank_biserial(u1, x.size, y.size), n1=x.size, n2=y.size)


def cell_confidence_intervals(df, rng):
    rows = []
    for (algo, maze, pop), cell in df.groupby(["algo", "maze_type", "pop_size"]):
        for metric in ALL_RUN_METRICS + SUCCESS_ONLY + SPARSE:
            mean, lo, hi, n = bootstrap_ci(metric_values(cell, metric), rng)
            rows.append(dict(algo=algo, maze_type=maze, pop_size=pop, metric=metric,
                             n=n, mean=mean, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def pairwise_tests(df):
    rows = []
    for (maze, pop), cell in df.groupby(["maze_type", "pop_size"]):
        for metric in PAIRWISE_METRICS:
            vals = {a: metric_values(cell[cell.algo == a], metric) for a in ALGOS}
            for a, b in PAIRS:
                res = mann_whitney(vals[a], vals[b])
                row = dict(maze_type=maze, pop_size=pop, metric=metric,
                           algo_a=a, algo_b=b)
                if res is None:
                    row.update(u=np.nan, p_raw=np.nan, p_holm=np.nan,
                               rank_biserial=np.nan, n_a=vals[a].size, n_b=vals[b].size,
                               note="insufficient n")
                else:
                    row.update(u=res["u"], p_raw=res["p"], p_holm=np.nan,
                               rank_biserial=res["r"], n_a=res["n1"], n_b=res["n2"],
                               note="")
                rows.append(row)
    out = pd.DataFrame(rows)
    # Holm correction within each metric, across all maze x pop cells and pairs.
    for metric in PAIRWISE_METRICS:
        mask = (out.metric == metric) & out.p_raw.notna()
        if mask.any():
            out.loc[mask, "p_holm"] = multipletests(out.loc[mask, "p_raw"], method="holm")[1]
    return out


def factor_effects(df):
    """Population main effect and algorithm x population interaction per metric."""
    rows = []
    full = df.copy()
    full["success_overall"] = full["success_overall"].astype(int)
    for metric in ALL_RUN_METRICS:
        model = ols(f"{metric} ~ C(algo) * C(maze_type) * C(pop_size)", data=full).fit()
        table = anova_lm(model, typ=2)
        for label, key in [("population main effect", "C(pop_size)"),
                           ("algorithm x population", "C(algo):C(pop_size)")]:
            rows.append(dict(metric=metric, scope="all mazes", effect=label,
                             F=table.loc[key, "F"], p=table.loc[key, "PR(>F)"]))
    # Optimality is only informative on Parallel Paths successful runs.
    pp = df[(df.maze_type == "Parallel Paths") & df.success_overall]
    if len(pp) > MIN_GROUP:
        model = ols("path_optimality ~ C(algo) * C(pop_size)", data=pp).fit()
        table = anova_lm(model, typ=2)
        for label, key in [("population main effect", "C(pop_size)"),
                           ("algorithm x population", "C(algo):C(pop_size)")]:
            if key in table.index:
                rows.append(dict(metric="path_optimality", scope="Parallel Paths",
                                 effect=label, F=table.loc[key, "F"], p=table.loc[key, "PR(>F)"]))
    return pd.DataFrame(rows)


def write_report(ci, pairwise, factors, path, input_path, n_rows):
    L = []
    L.append("Statistical analysis")
    L.append("=" * 70)
    L.append(f"Input: {input_path}   Runs: {n_rows}")
    L.append("Bootstrap: 10,000 resamples, 95% percentile intervals.")
    L.append("Pairwise: two-sided Mann-Whitney U, Holm correction within each metric,")
    L.append("rank-biserial effect size (positive = first algorithm tends higher).")
    L.append("")

    L.append("1. Per-cell means with 95% bootstrap CIs")
    L.append("-" * 70)
    for metric in ALL_RUN_METRICS + SUCCESS_ONLY + SPARSE:
        L.append(f"\n[{metric}]")
        sub = ci[ci.metric == metric]
        for _, r in sub.iterrows():
            L.append(f"  {r.algo:<4} {r.maze_type:<20} pop={int(r.pop_size):<4} "
                     f"n={int(r.n):<4} mean={r['mean']:.3f}  "
                     f"95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]")

    L.append("\n\n2. Pairwise Mann-Whitney U (Holm-significant, p_holm < 0.05)")
    L.append("-" * 70)
    sig = pairwise[(pairwise.p_holm < 0.05)].sort_values(["metric", "maze_type", "pop_size"])
    if sig.empty:
        L.append("  none")
    for _, r in sig.iterrows():
        L.append(f"  {r.metric:<20} {r.maze_type:<18} pop={int(r.pop_size):<4} "
                 f"{r.algo_a} vs {r.algo_b}: p_holm={r.p_holm:.2e}  r={r.rank_biserial:+.2f}")
    L.append(f"\n  ({len(sig)} of {pairwise.p_raw.notna().sum()} tests significant after Holm; "
             "full table in statistical_analysis_pairwise.csv)")

    L.append("\n\n3. Factor effects (Type II ANOVA)")
    L.append("-" * 70)
    for _, r in factors.iterrows():
        sig = "*" if r.p < 0.05 else " "
        L.append(f"  {r.metric:<18} {r.scope:<16} {r.effect:<24} "
                 f"F={r.F:8.2f}  p={r.p:.2e} {sig}")

    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Run the report's statistical analysis plan.")
    ap.add_argument("--input", default="results/runs.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df = pd.read_csv(args.input)

    ci = cell_confidence_intervals(df, rng)
    pairwise = pairwise_tests(df)
    factors = factor_effects(df)

    ci.to_csv(f"{args.outdir}/statistical_analysis_cis.csv", index=False)
    pairwise.to_csv(f"{args.outdir}/statistical_analysis_pairwise.csv", index=False)
    write_report(ci, pairwise, factors, f"{args.outdir}/statistical_analysis.txt",
                 args.input, len(df))
    print(f"Wrote statistical_analysis.txt, _cis.csv, _pairwise.csv to {args.outdir}/")


if __name__ == "__main__":
    main()
