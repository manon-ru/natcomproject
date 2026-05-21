"""
TDD RED tests for ACO pheromone evaporation timing, formula, and bounds.

Per Dorigo 1996 AS, evaporation follows:
    τᵢⱼ(t+1) = (1 - ρ) · τᵢⱼ(t) + Δτᵢⱼ(t)

where:
    ρ  = evaporation rate (ρ ∈ (0, 1])
    Δτ = Q/Lk if ant k completed path via cell (i,j), else 0

The project uses a per-cell pheromone floor of 0.01 (see aco.py evaporate()).
Initial pheromone τ₀ = 0.8, per Akka & Khaber (2018).

Most evaporation tests PASS on current aco.py (evaporation formula is correct).
test_evaporation_formula_is_multiply_1_minus_rho is RED: current aco.py deposits
flat Q=2.0 on every forward step (line 144) rather than Q/L only for completed
paths, so visited cells exceed the pure-evaporation baseline of τ₀·(1−ρ) = 0.72.

References:
    Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization
    by a colony of cooperating agents. IEEE Transactions on Systems, Man, and
    Cybernetics, Part B, 26(1), 29–41. doi:10.1109/3477.484436

    Akka, A., & Khaber, F. (2018). Mobile robot path planning using an
    improved ant colony optimization. International Journal of Advanced
    Robotic Systems, 15(3). doi:10.1177/1729881418774673
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from algorithms.aco import ACO
from maze.environment import MazeEnvironment
from tests.fixtures.small_mazes import trivial_3x3


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


def test_pheromone_initialized_to_tau_zero():
    """Initial pheromone τ₀ = 0.8 on all cells, per Akka & Khaber 2018.

    Before any call to run(), the entire pheromone matrix must be uniformly
    equal to initial_pheromone (τ₀ = 0.8). This is the baseline from which
    the evaporation formula decays over iterations.
    """
    maze = trivial_3x3()
    aco = ACO(maze, initial_pheromone=0.8)
    # No run() call — checking construction state only
    assert np.allclose(aco.pheromones, 0.8), (
        f"Expected all pheromones == 0.8 after construction (τ₀ = 0.8). "
        f"Got min={aco.pheromones.min():.5f}, max={aco.pheromones.max():.5f}."
    )


def test_evaporation_formula_is_multiply_1_minus_rho():
    """τ(t+1) = (1-ρ)·τ(t) when no path is completed, per Dorigo 1996 AS.

    In a 20×20 open maze the goal sits 38 Manhattan steps from start. After
    exactly 1 iteration (1 step per ant), no ant can reach (19,19). Therefore
    only evaporation applies — no Q/L deposit — and every cell must equal
    exactly τ₀ · (1−ρ) = 0.8 · 0.9 = 0.72.

    NOTE: FAILS on current aco.py because it deposits flat Q=2.0 on every
    forward step (line 144) regardless of goal completion. Cells that ants
    moved to get pheromone ≈ 0.72 + 2.0 = 2.72 instead of 0.72.

    This is both an evaporation formula test AND a deposit ordering test:
    the formula τ·(1−ρ) is correct, but spurious per-step deposits violate
    the Dorigo 1996 rule that Δτ = 0 for incomplete paths.
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

    # Per Dorigo 1996 AS Q/L rule: no completed path → Δτ = 0.
    # All cells must be exactly evaporation-reduced: τ₀·(1−ρ) = 0.72.
    expected = tau_0 * (1.0 - rho)  # 0.72
    assert np.all(aco.pheromones <= expected + 1e-6), (
        f"Expected all pheromones ≤ {expected:.4f} after 1 iteration "
        f"(τ₀·(1−ρ) = {tau_0}·{1.0 - rho} = {expected:.4f}, evaporation only). "
        f"Max observed = {aco.pheromones.max():.5f}. "
        "Per Dorigo 1996 AS: incomplete paths have Δτ = 0; flat per-step "
        f"deposit Q={2.0} inflates cells ants visited above evap-only baseline."
    )


def test_pheromone_floor_is_001():
    """Project-specific pheromone floor of 0.01 prevents numerical underflow.

    After enough iterations for natural decay to drop below 0.01
    (0.8 · 0.9^n < 0.01 requires n > 42), the floor must clamp all values.
    Isolating the goal means no ant ever completes a path, so this tests
    pure evaporation → floor behaviour.

    Implemented in aco.py evaporate():
        self.pheromones = np.maximum(self.pheromones, 0.01)
    """
    maze = trivial_3x3()
    # Block the goal cell entirely so no ant can ever reach (2,2)
    maze.add_wall((1, 2), (2, 2))
    maze.add_wall((2, 1), (2, 2))

    aco = ACO(
        maze,
        num_ants=1,
        evaporation_rate=0.1,
        pheromone_deposit=2.0,
        initial_pheromone=0.8,
    )
    aco.run(max_iterations=100)  # far past the ~42-iteration floor threshold

    assert np.all(aco.pheromones >= 0.01), (
        f"Pheromone floor 0.01 violated. "
        f"Min observed = {aco.pheromones.min():.8f}. "
        "evaporate() must enforce: pheromones = np.maximum(pheromones, 0.01) "
        "to prevent numerical underflow after extended evaporation."
    )


def test_pheromone_never_negative():
    """After any number of iterations, all pheromone values remain ≥ 0.

    Multiplication by (1−ρ) with ρ ∈ (0,1) preserves sign. The 0.01 floor
    further guarantees positivity. This test uses aggressive evaporation
    (ρ=0.5) over 200 iterations to stress-test the floor mechanism.
    """
    maze = trivial_3x3()
    aco = ACO(
        maze,
        num_ants=5,
        evaporation_rate=0.5,  # aggressive: halve pheromone each iteration
        pheromone_deposit=2.0,
        initial_pheromone=0.8,
    )
    aco.run(max_iterations=200)

    assert np.all(aco.pheromones >= 0), (
        f"Negative pheromone value detected. "
        f"Min observed = {aco.pheromones.min():.8f}. "
        "τ·(1−ρ) with ρ<1 and floor=0.01 must keep all pheromones ≥ 0 "
        "regardless of evaporation rate or iteration count."
    )


def test_evaporation_before_deposit_in_iteration():
    """Evaporation happens before deposit within each iteration.

    For a 2-cell maze, one ant can reach the goal in a single move. The
    canonical update is evaporation first, then deposit on the reached cell:

        τ' = τ₀·(1−ρ) + Q/L

    With τ₀=0.8, ρ=0.1, Q=2.0, L=2, expected goal-cell pheromone is 1.72.
    Current aco.py deposits flat Q per step, so this assertion is RED.
    """
    maze = MazeEnvironment(2, 1, (0, 0), (1, 0))
    maze.remove_wall((0, 0), (1, 0))
    aco = ACO(maze, num_ants=1, evaporation_rate=0.1, pheromone_deposit=2.0)

    aco.run(max_iterations=1)

    expected_goal = 0.8 * (1.0 - 0.1) + (2.0 / 2.0)
    assert np.isclose(aco.pheromones[0, 1], expected_goal)
