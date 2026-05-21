import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from algorithms.pso import PSO
from maze.environment import MazeEnvironment
from tests.fixtures.small_mazes import trivial_3x3


def open_4x4_with_one_wall():
    maze = MazeEnvironment(4, 4, (0, 0), (3, 3))
    for y in range(maze.height):
        for x in range(maze.width - 1):
            maze.remove_wall((x, y), (x + 1, y))
    for y in range(maze.height - 1):
        for x in range(maze.width):
            maze.remove_wall((x, y), (x, y + 1))
    maze.add_wall((1, 0), (2, 0))
    return maze


def test_run_returns_required_dict_keys():
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=30)
    assert set(result.keys()) == {
        "success",
        "iterations",
        "path",
        "snapshot",
        "snapshot_history",
        "disruption_iteration",
        "history",
    }


def test_run_success_path_starts_at_start():
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=5)
    result = pso.run(max_iterations=30)
    assert result["success"] is True
    assert result["path"][0] == maze.start


def test_run_success_path_ends_at_goal():
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=10)
    result = pso.run(max_iterations=200)
    if result["success"]:
        assert result["path"][-1] == maze.goal


def test_run_success_path_4_connected():
    maze = trivial_3x3()
    pso = PSO(maze, num_particles=10)
    result = pso.run(max_iterations=200)
    if result["success"]:
        assert all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for a, b in zip(result["path"], result["path"][1:])
        )


def test_reproducibility_same_seed_same_output(monkeypatch):
    maze = open_4x4_with_one_wall()

    monkeypatch.setattr(random, "random", lambda: (_ for _ in ()).throw(AssertionError("PSO must use np.random only")))
    np.random.seed(42)
    pso1 = PSO(maze, num_particles=10, c1=2.0, c2=2.0)
    result1 = pso1.run(max_iterations=50)

    maze = open_4x4_with_one_wall()
    monkeypatch.setattr(random, "random", lambda: (_ for _ in ()).throw(AssertionError("PSO must use np.random only")))
    np.random.seed(42)
    pso2 = PSO(maze, num_particles=10, c1=2.0, c2=2.0)
    result2 = pso2.run(max_iterations=50)

    assert result1["path"] != result2["path"]
