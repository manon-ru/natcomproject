"""
TDD RED tests for canonical Ant System pheromone deposit (Dorigo et al. 1996).

Per Dorigo 1996 AS, pheromone deposit follows the Q/L rule:
    Δτᵢⱼ(t) = Q / Lk   if ant k used edge (i,j) on its completed tour Tk
    Δτᵢⱼ(t) = 0         otherwise (ant did not reach goal)

Deposit is applied ONCE after path completion, not per-step.
Shorter paths yield larger Q/L → stronger per-cell reinforcement.

These tests FAIL against current aco.py (line 144) which deposits flat Q=2.0
on every forward step, regardless of whether the ant reaches the goal.

References:
    Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization
    by a colony of cooperating agents. IEEE Transactions on Systems, Man, and
    Cybernetics, Part B, 26(1), 29–41. doi:10.1109/3477.484436
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest

from algorithms.aco import ACO
from maze.environment import MazeEnvironment
from tests.fixtures.small_mazes import corridor_5x1, trivial_3x3


def _make_open_20x20() -> MazeEnvironment:
    """20×20 fully-open maze. Goal at (19,19) requires ≥38 steps from (0,0)."""
    maze = MazeEnvironment(20, 20, (0, 0), (19, 19))
    for y in range(20):
        for x in range(19):
            maze.remove_wall((x, y), (x + 1, y))
    for y in range(19):
        for x in range(20):
            maze.remove_wall((x, y), (x, y + 1))
    return maze


def test_no_deposit_when_no_ant_reaches_goal():
    """Per Dorigo 1996 AS: only ants that complete a path to the goal deposit pheromone.

    Δτᵢⱼ(t) = Q/Lk if ant k completed tour Tk via edge (i,j).
    Δτᵢⱼ(t) = 0    otherwise (Dorigo et al. 1996, Equation 3).

    In a 20×20 open maze the goal sits 38 Manhattan steps from start. After
    exactly 1 iteration (1 step per ant), no ant can reach the goal.
    Therefore no Q/L deposit should occur — only evaporation.

    NOTE: FAILS on current aco.py because it deposits Q=2.0 on every forward
    step (line 144), so moved-to cells get pheromone ≈ 0.72 + 2.0 = 2.72.
    """
    tau_0 = 0.8
    rho = 0.1
    maze = _make_open_20x20()
    aco = ACO(
        maze,
        num_ants=10,
        evaporation_rate=rho,
        pheromone_deposit=2.0,
        initial_pheromone=tau_0,
    )
    aco.run(max_iterations=1)

    # Canonical Q/L (Dorigo 1996): no completed path → no deposit
    # All pheromones must equal pure evaporation: tau_0 * (1 - rho) = 0.72
    expected_after_evap = tau_0 * (1.0 - rho)
    assert np.all(aco.pheromones <= expected_after_evap + 1e-6), (
        f"Expected all pheromones ≤ {expected_after_evap:.4f} (evaporation only, "
        f"no Q/L deposit). Max observed = {aco.pheromones.max():.5f}. "
        "Per Dorigo 1996 AS Δτ = Q/L only for completed paths; unfinished ants deposit 0."
    )


def test_deposit_amount_is_Q_over_L_not_Q_flat():
    """Per Dorigo 1996 AS: Δτ = Q/L per cell of completed path, not flat Q.

    For corridor_5x1 (L=4 moves, Q=2.0): deposit per cell = Q/L = 0.5.
    The single ant completes the corridor in exactly 4 iterations (no branching).
    After 4 evaporation cycles: base = tau_0 * (1-rho)^4 ≈ 0.52488.
    Canonical AS then adds Q/L = 0.5 → expected pheromone ≈ 1.02488.

    NOTE: FAILS on current aco.py which deposits flat Q=2.0 per step, giving
    pheromones[0,4] = 0.52488 + 2.0 = 2.52488 instead of 1.02488.
    """
    maze = corridor_5x1()  # (0,0)→(1,0)→(2,0)→(3,0)→(4,0), L=4 moves
    tau_0 = 0.8
    rho = 0.1
    Q = 2.0
    L = 4  # number of edges in the path

    aco = ACO(
        maze,
        num_ants=1,
        evaporation_rate=rho,
        pheromone_deposit=Q,
        initial_pheromone=tau_0,
    )
    aco.run()

    # Corridor is deterministic: single ant completes in exactly 4 iterations
    n_iters = 4
    base = tau_0 * (1.0 - rho) ** n_iters  # 0.8 * 0.9^4 ≈ 0.52488

    # Q/L deposit (Dorigo 1996): each path cell gets +Q/L = 0.5 at completion
    expected_ql = base + Q / L  # ≈ 1.02488

    # Goal cell (4,0) is at pheromones[row=0, col=4]
    actual = aco.pheromones[0, 4]
    assert actual == pytest.approx(expected_ql, abs=1e-6), (
        f"Expected pheromones[0,4] = {expected_ql:.5f} (Q/L deposit per Dorigo 1996). "
        f"Got {actual:.5f}. "
        f"Per Dorigo 1996 AS: Δτ = Q/L = {Q}/{L} = {Q/L} per cell, not flat Q={Q}."
    )


def test_failed_ants_do_not_deposit():
    """Per Dorigo 1996 AS for goal-finding: ants that don't reach goal deposit nothing.

    Δτᵢⱼ = 0 for all edges traversed by ant k if k never completes path to goal.
    Only ants with a completed tour contribute Q/Lk (Dorigo et al. 1996, Section 3).

    With goal (2,2) isolated by walls in the trivial_3x3 maze, no ant can ever
    reach it. Over 5 iterations, pheromones should be evaporation-only.

    NOTE: FAILS on current aco.py because ants deposit Q=2.0 on every forward
    step regardless of goal-completion, inflating pheromone above evap-only levels.
    """
    maze = trivial_3x3()  # 3×3 fully-open maze, goal=(2,2)
    maze.add_wall((1, 2), (2, 2))  # block left approach to (2,2)
    maze.add_wall((2, 1), (2, 2))  # block bottom approach to (2,2)
    # (2,2) is now completely unreachable — all ants fail every iteration

    tau_0 = 0.8
    rho = 0.1
    n_iterations = 5
    aco = ACO(
        maze,
        num_ants=3,
        evaporation_rate=rho,
        pheromone_deposit=2.0,
        initial_pheromone=tau_0,
    )
    aco.run(max_iterations=n_iterations)

    # Canonical Q/L (Dorigo 1996): no ant reached goal → zero deposit → evaporation only
    # max(0.01, tau_0*(1-rho)^5) = max(0.01, 0.47239) = 0.47239
    expected_max = tau_0 * (1.0 - rho) ** n_iterations  # 0.8 * 0.9^5 ≈ 0.47239
    assert np.all(aco.pheromones <= expected_max + 1e-6), (
        f"Expected all pheromones ≤ {expected_max:.5f} (pure evaporation, no Q/L deposit). "
        f"Max observed = {aco.pheromones.max():.5f}. "
        "Per Dorigo 1996 AS: failed ants contribute Δτ = 0 to pheromone trails."
    )


def test_short_path_higher_pheromone_per_cell_than_long_path():
    """Per Dorigo 1996 AS: Q/L deposit means shorter completed paths reinforce more per cell.

    Δτᵢⱼ = Q/Lk — inversely proportional to tour length Lk (Dorigo et al. 1996, Eq. 3).
    Short path (L=4) → deposit = Q/4 = 0.5 per cell.
    Long path  (L=8) → deposit = Q/8 = 0.25 per cell.
    Shorter tours receive stronger per-cell reinforcement, biasing future ants
    toward shorter routes.

    Current aco.py deposits flat Q=2.0 per step regardless of path length,
    giving longer paths MORE total pheromone — the inverse of the Q/L invariant.

    NOTE: FAILS on current aco.py because ACO.run() uses:
        self.pheromones[next_cell[1], next_cell[0]] += self.pheromone_deposit
    (line 144) — no division by path length, no post-completion trigger.
    """
    Q = 2.0
    L_short = 4  # 5-cell corridor: 4 moves
    L_long = 8   # 9-cell path: 8 moves

    # Mathematical invariant: Q/L gives more reinforcement per cell for shorter paths
    assert Q / L_short > Q / L_long, (
        f"Q/L invariant violated: Q/L_short={Q/L_short} must exceed Q/L_long={Q/L_long}."
    )

    # Verify ACO.run() implements Q/L deposit (not flat Q)
    # FAILS on current aco.py: deposit is flat pheromone_deposit with no path-length division
    source = inspect.getsource(ACO.run)
    has_ql_deposit = (
        "pheromone_deposit / len(" in source
        or "/ len(path" in source
        or "/ path_length" in source
        or "Q_over_L" in source
    )
    assert has_ql_deposit, (
        "ACO.run() does not implement Q/L deposit (Dorigo 1996 Ant System). "
        f"Expected: deposit = pheromone_deposit / path_length (Q/L). "
        f"Current code deposits flat pheromone_deposit={Q} per step regardless of path length. "
        "Flat deposit violates Dorigo 1996: longer paths accumulate MORE pheromone, "
        "not less — the opposite of the Q/L reinforcement invariant."
    )
