#!/bin/bash
# Run all QA scenarios and report pass/fail.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVIDENCE_DIR="$PROJECT_DIR/.sisyphus/evidence"
mkdir -p "$EVIDENCE_DIR"

PASS=0
FAIL=0

run_qa() {
    local name="$1"
    local cmd="$2"
    echo -n "  $name ... "
    if eval "$cmd" > "$EVIDENCE_DIR/qa-${name}.txt" 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        cat "$EVIDENCE_DIR/qa-${name}.txt"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== QA Suite ==="
cd "$PROJECT_DIR"

run_qa "config"         "uv run python -c \"import sys; sys.path.insert(0,'src'); from config import POPULATION_SIZES, ACO_PARAMS, PSO_PARAMS, GA_PARAMS, DISRUPTION_TIME, RECOVERY_THRESHOLD; assert POPULATION_SIZES == [20, 50, 150]; assert ACO_PARAMS['beta'] == 5; assert PSO_PARAMS['omega'] == 1.0; assert GA_PARAMS['mutation_rate'] == 0.3; assert DISRUPTION_TIME == 100; assert RECOVERY_THRESHOLD == 0.8; print('OK')\""
run_qa "es_gone"        "test ! -f src/algorithms/es.py"
run_qa "cleanup"        "! grep -q 'from turtle' src/maze/generator.py && ! grep -q 'Zorg dat' src/maze/generator.py && echo OK"
run_qa "pso_omega"      "uv run python -c \"import sys; sys.path.insert(0,'src'); from maze.generator import generate_maze; from algorithms.pso import PSO; m = generate_maze(20,20,seed=1,maze_type='U-Trap'); p = PSO(m); assert p.omega == 1.0; print('OK')\""
run_qa "aco_params"     "uv run python -c \"import sys; sys.path.insert(0,'src'); from maze.generator import generate_maze; from algorithms.aco import ACO; import numpy as np; m = generate_maze(20,20,seed=1,maze_type='U-Trap'); a = ACO(m); assert a.beta == 5.0 and a.pheromone_deposit == 2.0 and np.allclose(a.pheromones, 0.8); print('OK')\""
run_qa "ga_path_encoding" "uv run python scripts/qa/qa_ga_path_encoding.py"
run_qa "forced_min"     "uv run python scripts/qa/qa_forced_min_iterations.py"
run_qa "reproducibility" "uv run python scripts/qa/qa_reproducibility.py"
run_qa "metric_fix"     "uv run python -c \"import sys; sys.path.insert(0,'src'); from evaluation.metrics import time_to_half_entropy; h=[0.5,1.0,0.9,0.7,0.4,0.3,0.2]; assert time_to_half_entropy(h,10)==30; print('OK')\""
run_qa "adaptation_floor" "uv run python -c \"import sys; sys.path.insert(0,'src'); from evaluation.metrics import adaptation_time; h=[1.0,0.05,0.05,0.3,0.5,0.8,1.0]; assert adaptation_time(h,20,sample_interval=10,entropy_floor=0.1) is None; print('OK')\""

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ $FAIL -eq 0 ]; then
    echo "ALL_PASS"
    exit 0
else
    echo "SOME_FAILED"
    exit 1
fi
