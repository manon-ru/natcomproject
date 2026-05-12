"""QA: GA two-point crossover produces expected child on fixed inputs."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm

def main():
    m = generate_maze(20, 20, seed=1, maze_type="U-Trap")
    ga = GeneticAlgorithm(m, pop_size=2, crossover_rate=1.0)  # always crossover
    L = ga.chromosome_length

    chrom_a = [0] * L
    chrom_b = [3] * L
    parent_a = {"chromosome": chrom_a, "executed_path": [], "end_position": m.start, "reached_goal": False}
    parent_b = {"chromosome": chrom_b, "executed_path": [], "end_position": m.start, "reached_goal": False}

    # Fix crossover points via seeded random
    random.seed(42)
    child = ga._crossover(parent_a, parent_b)
    chrom = child["chromosome"]

    # Verify: child is a valid combination of parent_a and parent_b segments
    assert len(chrom) == L, f"wrong length: {len(chrom)}"
    assert all(g in (0, 3) for g in chrom), f"unexpected gene values: {set(chrom)}"

    # Verify it's not all-zeros or all-threes (crossover happened)
    has_zeros = any(g == 0 for g in chrom)
    has_threes = any(g == 3 for g in chrom)
    assert has_zeros and has_threes, "crossover didn't mix parents"

    print("OK crossover")
    sys.exit(0)

if __name__ == "__main__":
    main()
