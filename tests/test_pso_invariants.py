import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from algorithms.pso import PSO
from tests.fixtures.small_mazes import simple_choice_3x3, trivial_3x3


def make_pso(maze, num_particles=40):
    return PSO(maze, num_particles=num_particles)


def seed_everything(seed=7):
    random.seed(seed)
    np.random.seed(seed)


def run_manual_swarm(pso, iterations, disruption_iteration=-1):
    particles = pso.initialize_particles()
    global_best = [pso.maze.start]
    pbest_history = [[] for _ in particles]
    gbest_history = []

    for _ in range(iterations):
        for index, particle in enumerate(particles):
            pso.update_particle(particle, global_best, disruption_iteration=disruption_iteration)
            pbest_history[index].append(pso.distance_to_goal(particle["personal_best"]))
            if pso.distance_to_goal(particle["path"]) < pso.distance_to_goal(global_best):
                global_best = particle["path"][:]
        gbest_history.append(pso.distance_to_goal(global_best))

    return particles, pbest_history, gbest_history, global_best


def test_pbest_monotonic_per_particle():
    """Per standard PSO convergence (Clerc & Kennedy 2002): pbest distance to goal is non-increasing."""
    seed_everything(11)
    pso = make_pso(trivial_3x3(), num_particles=24)
    _, pbest_history, _, _ = run_manual_swarm(pso, iterations=50)

    for history in pbest_history:
        assert all(b <= a for a, b in zip(history, history[1:]))


def test_gbest_monotonic():
    """Per standard PSO: global best distance to goal is non-increasing."""
    seed_everything(13)
    pso = make_pso(trivial_3x3(), num_particles=24)
    _, _, gbest_history, _ = run_manual_swarm(pso, iterations=50)

    assert all(b <= a for a, b in zip(gbest_history, gbest_history[1:]))


def test_entropy_history_length_correct():
    """Entropy sampled every 10 iters: len(entropy_history) == floor(max_iters/10) + 1 for iters 0,10,...100."""
    seed_everything(21)
    pso = make_pso(trivial_3x3(), num_particles=30)
    pso.run(max_iterations=100)
    assert len(pso.entropy_history) == 11


def test_entropy_history_non_degenerate():
    seed_everything(22)
    pso = make_pso(trivial_3x3(), num_particles=30)
    pso.run(max_iterations=100)
    assert max(pso.entropy_history) > 0
    assert not any(math.isnan(value) for value in pso.entropy_history)


def test_no_import_random():
    """PSO must use np.random exclusively, not stdlib random."""
    source = (Path(__file__).resolve().parent.parent / "src" / "algorithms" / "pso.py").read_text()
    assert "import random" not in source


def test_disruption_no_freeze_at_goal_pre_disruption():
    """Particles that reach goal before disruption must continue updating (match GA semantics)."""
    seed_everything(31)
    maze = simple_choice_3x3()
    pso = make_pso(maze, num_particles=120)
    particles = pso.initialize_particles()
    global_best = [pso.maze.start]
    goal_length = len(pso.maze.shortest_path())
    hit_before_disruption = {}

    for iteration in range(100):
        for particle in particles:
            pso.update_particle(particle, global_best, disruption_iteration=50)
            if pso.distance_to_goal(particle["path"]) < pso.distance_to_goal(global_best):
                global_best = particle["path"][:]

            if iteration < 50 and particle["path"][-1] == pso.maze.goal:
                hit_before_disruption[id(particle)] = len(particle["path"])

    assert hit_before_disruption
    assert any(
        len(particle["path"]) > hit_before_disruption[id(particle)]
        for particle in particles
        if id(particle) in hit_before_disruption and len(particle["path"]) > goal_length
    )
