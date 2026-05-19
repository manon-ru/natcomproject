#!/bin/bash
set -euo pipefail

# Wait for the main 2700-run experiment to finish, then regenerate the three
# downstream artifacts in one shot: per-cell aggregate, figures pack, and
# hypothesis report. Idempotent - safe to re-run.
#
# Usage:
#   bash scripts/finalize_run.sh          # poll until done, then regenerate
#   bash scripts/finalize_run.sh --now    # skip waiting (use whatever's in runs.csv)

cd "$(dirname "$0")/.."

RUNS_CSV="results/runs.csv"
PID_FILE="results/run.pid"
EXPECTED_LINES=2701   # 2700 results + 1 header

if [[ "${1:-}" != "--now" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    echo "Watching experiment PID $PID; expecting $EXPECTED_LINES lines in $RUNS_CSV..."
    while kill -0 "$PID" 2>/dev/null; do
      LINES=$(wc -l < "$RUNS_CSV" 2>/dev/null || echo 0)
      printf "\r  PID $PID alive | $LINES / $EXPECTED_LINES rows"
      sleep 30
    done
    echo
    echo "Process $PID exited."
  fi
fi

FINAL_LINES=$(wc -l < "$RUNS_CSV")
echo "Final row count: $FINAL_LINES"

echo "--- 1/3: aggregate_summary ---"
uv run python scripts/aggregate_summary.py --input "$RUNS_CSV" --output results/aggregate_summary.txt

echo "--- 2/3: figures pack ---"
uv run python scripts/figures.py --input "$RUNS_CSV" --output figures/

echo "--- 3/3: hypothesis report ---"
uv run python scripts/hypothesis_report.py --input "$RUNS_CSV" --output results/hypothesis_report.txt

echo
echo "Done. Outputs:"
echo "  - results/aggregate_summary.txt"
echo "  - results/hypothesis_report.txt"
echo "  - figures/*.png  +  figures/summary_table.tex"
