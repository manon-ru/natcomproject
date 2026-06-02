#!/bin/bash
#SBATCH --job-name=natcom-g27
#SBATCH --partition=csedu
#SBATCH --account=cseduimc042
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=results/cluster-%j.out
#SBATCH --error=results/cluster-%j.err

set -e

mkdir -p results

# ── Python environment ───────────────────────────────────────────────────────
# Try uv first; fall back to a plain venv + pip install
if command -v uv &>/dev/null; then
    echo "[setup] Using uv"
    uv sync --quiet
    PYTHON="uv run python"
else
    echo "[setup] uv not found — creating venv"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet numpy matplotlib tqdm
    PYTHON="python"
fi

# ── Run experiment ───────────────────────────────────────────────────────────
echo "[start] $(date) — workers=${SLURM_CPUS_PER_TASK}"
$PYTHON main.py --workers=${SLURM_CPUS_PER_TASK}
echo "[done]  $(date)"
