"""QA: forced_min_iterations is honored — GA on Sudden Wall runs >=300 iterations."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from runner import run_single_trial

def main():
    result = run_single_trial(("GA", "Sudden Wall", 20, 1, 42))
    iters = result["iterations"]
    if iters >= 300:
        print(f"OK iterations={iters}>=300")
        sys.exit(0)
    else:
        print(f"FAIL iterations={iters} < 300 (forced_min not honored)")
        sys.exit(1)

if __name__ == "__main__":
    main()
