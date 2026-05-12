"""
visualize_run.py — Run a single trial of one algorithm and visualize the result.

Useful for inspecting algorithm behavior, especially when investigating why
an algorithm fails (e.g., GA 0% success on certain maze types).

Usage:
    uv run python scripts/visualize_run.py --algo GA --maze U-Trap
    uv run python scripts/visualize_run.py --algo PSO --maze "Sudden Wall" --pop 50
    uv run python scripts/visualize_run.py --algo ACO --maze "Parallel Paths" --pop 150
    uv run python scripts/visualize_run.py --algo GA --maze U-Trap --small
    uv run python scripts/visualize_run.py --algo GA --maze U-Trap --no-entropy

Each run displays:
  - Maze with walls, optimal path (red), best found path (green if any), explored cells (faint blue)
  - Entropy curve over iterations
  - Diagnostic statistics
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import MAZE_WIDTH, MAZE_HEIGHT, DISRUPTION_TIME
from maze.generator import generate_maze
from runner import _build_algorithm, _seed_worker
from visualization.plot import visualize_maze, visualize_entropy


def main():
    ap = argparse.ArgumentParser(description="Visualize a single algorithm trial.")
    ap.add_argument("--algo", choices=["GA", "PSO", "ACO"], default="GA")
    ap.add_argument("--maze", choices=["U-Trap", "Sudden Wall", "Parallel Paths"], default="U-Trap")
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1, help="Maze instance seed")
    ap.add_argument("--trial-seed", type=int, default=42, help="Algorithm trial seed")
    ap.add_argument("--iterations", type=int, default=None,
                    help="Max iterations (default: 10 × optimal_path_length, with disruption padding for Sudden Wall)")
    ap.add_argument("--small", action="store_true", help="Use 20×20 maze instead of 40×40 (faster, easier for GA)")
    ap.add_argument("--no-entropy", action="store_true", help="Skip the entropy curve plot")
    ap.add_argument("--no-maze-plot", action="store_true", help="Skip the maze visualization (diagnostics only)")
    args = ap.parse_args()

    W = 20 if args.small else MAZE_WIDTH
    H = 20 if args.small else MAZE_HEIGHT

    maze = generate_maze(W, H, seed=args.seed, maze_type=args.maze)
    initial_optimal_path = maze.shortest_path()
    if initial_optimal_path is None:
        print(f"ERROR: maze unreachable (maze={args.maze}, seed={args.seed})")
        sys.exit(1)
    initial_optimal_steps = len(initial_optimal_path) - 1

    true_optimal_steps = initial_optimal_steps
    true_optimal_path = initial_optimal_path
    if args.maze == "Sudden Wall" and getattr(maze, "dynamic_wall", None):
        w1, w2 = maze.dynamic_wall
        maze.add_wall(w1, w2)
        post_path = maze.shortest_path()
        if post_path is not None:
            true_optimal_path = post_path
            true_optimal_steps = len(post_path) - 1
        maze.remove_wall(w1, w2)

    print(f"═══ Single-trial visualization ═══")
    print(f"  Maze type:        {args.maze}")
    print(f"  Maze dimensions:  {W} × {H}")
    print(f"  Instance seed:    {args.seed}")
    print(f"  Trial seed:       {args.trial_seed}")
    print(f"  Pre-disruption optimal:  {initial_optimal_steps} cells")
    if args.maze == "Sudden Wall":
        print(f"  Post-disruption optimal: {true_optimal_steps} cells")
        print(f"  Disruption at iteration: {DISRUPTION_TIME}")

    if args.iterations is None:
        max_iters = max(true_optimal_steps * 10, 200)
        if args.maze == "Sudden Wall":
            max_iters = max(max_iters, DISRUPTION_TIME + 200)
    else:
        max_iters = args.iterations
    print(f"  Max iterations:   {max_iters}")
    print()

    _seed_worker(args.trial_seed)
    algo = _build_algorithm(args.algo, maze, args.pop)
    disruption = DISRUPTION_TIME if args.maze == "Sudden Wall" else -1
    forced_min = DISRUPTION_TIME + 200 if args.maze == "Sudden Wall" else 0

    print(f"═══ Algorithm: {args.algo} (pop={args.pop}) ═══")
    if args.algo == "GA":
        print(f"  Chromosome length: {algo.chromosome_length}")
        print(f"  Crossover rate:    {algo.crossover_rate}")
        print(f"  Mutation rate:     {algo.mutation_rate} per gene")
        expected_mutations = algo.chromosome_length * algo.mutation_rate
        print(f"  Expected mutations per offspring: {expected_mutations:.1f}")
    elif args.algo == "PSO":
        print(f"  Inertia ω: {algo.omega}")
        print(f"  Cognitive c1: {algo.c1}, Social c2: {algo.c2}")
    elif args.algo == "ACO":
        print(f"  α={algo.alpha}, β={algo.beta}, Q={algo.pheromone_deposit}")
        print(f"  Evaporation ρ={algo.evaporation_rate}, τ₀={algo.initial_pheromone}")
    print()

    print(f"Running... (this may take a moment)")
    result = algo.run(
        max_iterations=max_iters,
        disruption_iteration=disruption,
        forced_min_iterations=forced_min,
    )

    print()
    print(f"═══ Result ═══")
    print(f"  Success:    {result['success']}")
    print(f"  Iterations: {result['iterations']}")
    if result.get("path"):
        path_len = len(result["path"]) - 1
        ratio = path_len / true_optimal_steps if true_optimal_steps > 0 else 0
        print(f"  Path length: {path_len} cells (optimal={true_optimal_steps}, ratio={ratio:.2f})")
    if hasattr(algo, "entropy_history") and algo.entropy_history:
        eh = algo.entropy_history
        print(f"  Entropy: peak={max(eh):.3f}, min={min(eh):.3f}, mean={sum(eh)/len(eh):.3f} bits")

    if args.algo == "GA" and not result["success"]:
        print()
        print(f"═══ GA Diagnostics (why did it fail?) ═══")
        if result.get("path"):
            best_path_len = len(result["path"]) - 1
            gx, gy = maze.goal
            end_x, end_y = result["path"][-1]
            mh_dist = abs(end_x - gx) + abs(end_y - gy)
            print(f"  Best individual reached: {result['path'][-1]}")
            print(f"  Goal: {maze.goal}")
            print(f"  Manhattan distance to goal: {mh_dist}")
            print(f"  Cells traversed: {best_path_len}")
            print(f"  → Random direction-string chromosomes rarely encode a valid")
            print(f"    path. Each gene hitting a wall = stay-in-place,")
            print(f"    so a 160-gene chromosome may make very little progress.")
        print()
        if algo.entropy_history:
            print(f"  Mean entropy across run: {sum(algo.entropy_history)/len(algo.entropy_history):.3f} bits")
            print(f"  → High entropy confirms H1: 'GA decoupled → maintains diversity'")
            print(f"  → But mutation rate {algo.mutation_rate} × {algo.chromosome_length} genes = "
                  f"{algo.mutation_rate * algo.chromosome_length:.0f} expected mutations per offspring,")
            print(f"    which is destructive — children mostly differ from useful parents.")

    if not args.no_maze_plot:
        print()
        print(f"Showing maze visualization (close window to continue)...")
        history = result.get("history") or []
        visualize_maze(
            maze,
            optimal_path=true_optimal_path,
            best_found_path=result["path"] if result["success"] else None,
            all_paths=[history] if history else None,
            title=f"{args.algo} on {args.maze} (pop={args.pop}, seed={args.seed}) — "
                  f"{'SUCCESS' if result['success'] else 'FAILED'}",
            show=True,
        )

    if not args.no_entropy and hasattr(algo, "entropy_history") and algo.entropy_history:
        print(f"Showing entropy curve (close window to finish)...")
        visualize_entropy(
            {args.algo: algo.entropy_history},
            sample_interval=10,
            title=f"{args.algo} Shannon entropy — {args.maze} (pop={args.pop})",
            show=True,
        )


if __name__ == "__main__":
    main()
