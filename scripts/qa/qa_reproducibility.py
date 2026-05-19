"""QA: Same (algo, maze, pop, instance_seed, trial_seed) produces identical results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from runner import run_single_trial

def main():
    task = ("PSO", "U-Trap", 20, 1, 42)
    r1 = run_single_trial(task)
    r2 = run_single_trial(task)

    if r1["entropy_history"] != r2["entropy_history"]:
        print(f"FAIL entropy_history mismatch")
        print(f"  r1 len={len(r1['entropy_history'])}, r2 len={len(r2['entropy_history'])}")
        sys.exit(1)

    if r1["iterations"] != r2["iterations"]:
        print(f"FAIL iterations mismatch: {r1['iterations']} != {r2['iterations']}")
        sys.exit(1)

    if r1["success_overall"] != r2["success_overall"]:
        print(f"FAIL success_overall mismatch")
        sys.exit(1)

    print(f"OK reproducibility iters={r1['iterations']} success={r1['success_overall']}")
    sys.exit(0)

if __name__ == "__main__":
    main()
