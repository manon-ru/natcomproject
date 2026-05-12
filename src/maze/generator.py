import random
import numpy as np
from maze.environment import MazeEnvironment

braid_factor = 0.1  # Adjust this to control how many extra paths are added

def _braid_maze(maze, x1, x2, y1, y2, factor):
    if factor <= 0: return
    for y in range(y1, y2):
        for x in range(x1, x2):
            for dx, dy in [(1, 0), (0, 1)]:
                nx, ny = x + dx, y + dy
                if x1 <= nx < x2 and y1 <= ny < y2:
                    if maze.has_wall_between((x, y), (nx, ny)):
                        if random.random() < factor:
                            maze.remove_wall((x, y), (nx, ny))

def _generate_dfs(maze, x1, x2, y1, y2):
    if x2 <= x1 or y2 <= y1: return
    start_node = (random.randint(x1, x2-1), random.randint(y1, y2-1))
    visited = {start_node}
    stack = [start_node]

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if x1 <= nx < x2 and y1 <= ny < y2 and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if neighbors:
            nx, ny = random.choice(neighbors)
            maze.remove_wall((x, y), (nx, ny))
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            stack.pop()

def generate_maze(width: int, height: int, seed: int = 2026, maze_type: str = "Random") -> MazeEnvironment:
    random.seed(seed)
    np.random.seed(seed)
    
    if maze_type in ["Parallel Paths", "Sudden Wall"]:
        start, goal = (width // 2, 0), (width // 2, height - 1)
    elif maze_type == "U-Trap":
        start, goal = (0, height // 2), (width - 1, height // 2)
    else:
        start, goal = (0, 0), (width - 1, height - 1)

    maze = MazeEnvironment(width, height, start=start, goal=goal)

    if maze_type == "Random":
        _generate_dfs(maze, 0, width, 0, height)
        _braid_maze(maze, 0, width, 0, height, braid_factor)
        
    elif maze_type == "U-Trap":
        mid_y = height // 2
        _generate_dfs(maze, 0, width, 0, mid_y)
        _braid_maze(maze, 0, width, 0, mid_y, braid_factor)
        
        _generate_dfs(maze, 0, width, mid_y, height)
        _braid_maze(maze, 0, width, mid_y, height, braid_factor)

        for x in range(width):
            maze.horizontal_walls[mid_y, x] = True

        maze.remove_wall((0, mid_y), (0, mid_y - 1))

        maze.add_wall((width - 1, mid_y), (width - 2, mid_y))
        if mid_y + 1 < height:
            maze.add_wall((width - 1, mid_y), (width - 1, mid_y + 1))
        
        maze.remove_wall((width - 1, mid_y), (width - 1, mid_y - 1))

        for x in range(width - 2):
            maze.remove_wall((x, mid_y), (x + 1, mid_y))

    elif maze_type == "Sudden Wall":
        mid_x = width // 2
        _generate_dfs(maze, 0, mid_x, 0, height)
        _braid_maze(maze, 0, mid_x, 0, height, braid_factor)
        
        _generate_dfs(maze, mid_x + 1, width, 0, height)
        _braid_maze(maze, mid_x + 1, width, 0, height, braid_factor)
        
        for y in range(height - 1):
            maze.remove_wall((mid_x, y), (mid_x, y + 1))
            if 0 < y < height - 1:
                maze.vertical_walls[y, mid_x] = True
                maze.vertical_walls[y, mid_x + 1] = True
            
        for side_x in [mid_x - 1, mid_x + 1]:
            maze.remove_wall((mid_x, 0), (side_x, 0))
            maze.remove_wall((mid_x, height - 1), (side_x, height - 1))
            
        maze.dynamic_wall = ((mid_x, height // 2), (mid_x, (height // 2) + 1))

    elif maze_type == "Parallel Paths":
        # FIX: Force an odd "effective" width so the center column perfectly balances parity
        effective_width = width - 1 
        mid_x = effective_width // 2 # Center is exactly 9
        
        maze.start = (mid_x, 0)
        maze.goal = (mid_x, height - 1)
        start, goal = maze.start, maze.goal

        # Step 1: Generate & braid left maze (cols 0 to 8)
        _generate_dfs(maze, 0, mid_x, 1, height - 2)
        _braid_maze(maze, 0, mid_x, 1, height - 2, braid_factor)

        # Step 2: Perfect Mirror left to right (cols 10 to 18)
        # Because we braid BEFORE mirroring, both sides receive the EXACT same loops.
        for y in range(1, height - 2):
            for x in range(mid_x):
                rx = (effective_width - 1) - x
                maze.horizontal_walls[y, rx] = maze.horizontal_walls[y, x]
                maze.horizontal_walls[y+1, rx] = maze.horizontal_walls[y+1, x]
                maze.vertical_walls[y, rx+1] = maze.vertical_walls[y, x]
                maze.vertical_walls[y, rx] = maze.vertical_walls[y, x+1]

        # Step 3: Seal the middle corridor (col 9) and the unused col 19
        for y in range(height):
            maze.vertical_walls[y, mid_x] = True
            maze.vertical_walls[y, mid_x + 1] = True
            maze.vertical_walls[y, effective_width] = True 
            maze.horizontal_walls[y, effective_width] = True
            if y + 1 < height: maze.horizontal_walls[y+1, effective_width] = True

        # Step 4: Connect Start (9,0) to both halves symmetrically
        maze.remove_wall(start, (mid_x - 1, 0))          # Left to 8,0
        maze.remove_wall((mid_x - 1, 0), (mid_x - 1, 1)) # Down to 8,1
        
        maze.remove_wall(start, (mid_x + 1, 0))          # Right to 10,0
        maze.remove_wall((mid_x + 1, 0), (mid_x + 1, 1)) # Down to 10,1
        
        maze.horizontal_walls[1, mid_x] = True           # Block going straight down from Start

        # Step 5: The Detours at the end!
        # Left exit (The optimal straight path: 3 steps)
        maze.remove_wall((mid_x - 1, height - 3), (mid_x - 1, height - 2))
        maze.remove_wall((mid_x - 1, height - 2), (mid_x - 1, height - 1))
        maze.remove_wall((mid_x - 1, height - 1), goal)

        # Right exit (The tricky detour: 5 steps)
        maze.remove_wall((mid_x + 1, height - 3), (mid_x + 1, height - 2)) 
        maze.remove_wall((mid_x + 1, height - 2), (mid_x + 2, height - 2)) 
        maze.remove_wall((mid_x + 2, height - 2), (mid_x + 2, height - 1)) 
        maze.remove_wall((mid_x + 2, height - 1), (mid_x + 1, height - 1)) 
        maze.remove_wall((mid_x + 1, height - 1), goal)
        
        # Block going straight down to the goal from the center
        maze.horizontal_walls[height - 1, mid_x] = True 

        # Step 6: Bottom random walls to create dead ends safely
        for _ in range(4):
            for y in range(height - 2, height):
                for x in range(effective_width):
                    if maze.horizontal_walls[y, x] and maze.horizontal_walls[y+1, x] and maze.vertical_walls[y, x] and maze.vertical_walls[y, x+1]:
                        neighbors = []
                        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < effective_width and 0 <= ny < height and not (x < mid_x and nx >= mid_x) and not (x > mid_x and nx <= mid_x):
                                if not (maze.horizontal_walls[ny, nx] and maze.horizontal_walls[ny+1, nx] and maze.vertical_walls[ny, nx] and maze.vertical_walls[ny, nx+1]):
                                    if (nx, ny) != goal:
                                        neighbors.append((nx, ny))
                        if neighbors:
                            maze.remove_wall((x, y), random.choice(neighbors))

    return maze