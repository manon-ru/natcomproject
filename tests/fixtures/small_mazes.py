import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from maze.environment import MazeEnvironment


def _open_all_adjacent_walls(maze: MazeEnvironment) -> None:
    for y in range(maze.height):
        for x in range(maze.width - 1):
            maze.remove_wall((x, y), (x + 1, y))
    for y in range(maze.height - 1):
        for x in range(maze.width):
            maze.remove_wall((x, y), (x, y + 1))


def trivial_3x3() -> MazeEnvironment:
    maze = MazeEnvironment(3, 3, (0, 0), (2, 2))
    _open_all_adjacent_walls(maze)
    return maze


def corridor_5x1() -> MazeEnvironment:
    maze = MazeEnvironment(5, 1, (0, 0), (4, 0))
    for x in range(4):
        maze.remove_wall((x, 0), (x + 1, 0))
    return maze


def simple_choice_3x3() -> MazeEnvironment:
    maze = MazeEnvironment(3, 3, (0, 0), (2, 2))
    _open_all_adjacent_walls(maze)
    maze.add_wall((1, 0), (1, 1))
    return maze
