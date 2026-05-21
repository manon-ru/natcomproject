"""QA: GA chromosomes are valid path sequences."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from maze.generator import generate_maze
from algorithms.ga import GeneticAlgorithm

m = generate_maze(40, 40, seed=1, maze_type="Shortest Path Trap")
ga = GeneticAlgorithm(m, pop_size=20)
random.seed(0)
pop = ga._initialize_population()
for i, ind in enumerate(pop):
    path = ind["path"]
    assert len(path) >= 1, f"empty path in individual {i}"
    assert path[0] == m.start, f"path doesn't start at maze.start in individual {i}"
    # Verify each consecutive pair is adjacent
    for a, b in zip(path[:-1], path[1:]):
        assert b in m.neighbors(*a), f"invalid step {a}->{b} in individual {i}"
    # Verify no revisits
    assert len(set(path)) == len(path), f"path has revisits in individual {i}"
print(f"OK path_encoding pop={len(pop)} avg_len={sum(len(ind['path']) for ind in pop) / len(pop):.1f}")
