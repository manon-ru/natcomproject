"""QA: GA roulette selection bias matches expected proportions (chi-square test)."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm

def main():
    random.seed(0)
    m = generate_maze(20, 20, seed=1, maze_type="U-Trap")
    ga = GeneticAlgorithm(m, pop_size=5)

    gx, gy = m.goal
    # dist=5 is the minimum-fitness dummy (shifted weight ≈0)
    # dist=4,3,2,1 get shifted weights ≈1,2,3,4 → proportions [1,2,3,4]/10
    population = []
    for dist in [5, 4, 3, 2, 1]:
        end = (gx - dist, gy) if gx - dist >= 0 else (gx, gy - dist)
        ind = {
            "chromosome": [0] * ga.chromosome_length,
            "executed_path": [m.start],
            "end_position": end,
            "reached_goal": False,
        }
        population.append(ind)

    N = 10000
    counts = [0] * 5
    for _ in range(N):
        selected = ga._roulette_select(population)
        idx = population.index(selected)
        counts[idx] += 1

    # Skip index 0 (dummy, expected weight ≈0); test indices 1-4 against [1,2,3,4]/10
    test_counts = counts[1:]
    expected_props = [1/10, 2/10, 3/10, 4/10]
    expected_counts = [p * N for p in expected_props]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(test_counts, expected_counts))

    critical_value = 11.345  # df=3, alpha=0.01

    if chi2 < critical_value:
        print(f"OK chi2={chi2:.3f} < {critical_value} (p>0.01)")
        sys.exit(0)
    else:
        print(f"FAIL chi2={chi2:.3f} >= {critical_value} (p<=0.01)")
        print(f"  counts={counts}, expected_props={expected_props}")
        sys.exit(1)

if __name__ == "__main__":
    main()
