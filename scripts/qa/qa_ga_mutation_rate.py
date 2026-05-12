"""QA: GA flip-bit mutation rate should be ~0.3 per gene."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm

def main():
    random.seed(0)
    m = generate_maze(20, 20, seed=1, maze_type="U-Trap")
    ga = GeneticAlgorithm(m, pop_size=1, mutation_rate=0.3)

    L = ga.chromosome_length
    total_genes = 0
    total_flips = 0
    N = 10000

    for _ in range(N):
        original = [random.randint(0, 3) for _ in range(L)]
        ind = {"chromosome": original[:], "executed_path": None, "end_position": None, "reached_goal": False}
        ga._mutate(ind)
        flips = sum(1 for a, b in zip(original, ind["chromosome"]) if a != b)
        total_genes += L
        total_flips += flips

    rate = total_flips / total_genes
    expected = 0.3
    tolerance = 0.02

    if abs(rate - expected) <= tolerance:
        print(f"OK rate={rate:.3f}")
        sys.exit(0)
    else:
        print(f"FAIL rate={rate:.3f} expected={expected}±{tolerance}")
        sys.exit(1)

if __name__ == "__main__":
    main()
