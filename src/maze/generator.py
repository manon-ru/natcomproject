import random
import numpy as np
from maze.environment import MazeEnvironment

def _generate_dfs(maze, x1, x2, y1, y2):
    """
    Generates a 'Perfect Maze' using randomized Depth-First Search.
    A perfect maze guarantees exactly one path between any two points.
    """
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
        
    elif maze_type == "U-Trap":
        mid_y = height // 2
        _generate_dfs(maze, 0, width, 0, mid_y)
        _generate_dfs(maze, 0, width, mid_y, height)

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
        _generate_dfs(maze, mid_x + 1, width, 0, height)
        
        for y in range(height - 1):
            maze.remove_wall((mid_x, y), (mid_x, y + 1))
            if 0 < y < height - 1:
                maze.vertical_walls[y, mid_x] = True
                maze.vertical_walls[y, mid_x + 1] = True
            
        # Connect BOTH wings at the top so they are both explored
        maze.remove_wall((mid_x, 0), (mid_x - 1, 0))
        maze.remove_wall((mid_x, 0), (mid_x + 1, 0))
        
        # Connect ONLY the left wing at the bottom. 
        # The right wing is now a massive dead-end trap.
        maze.remove_wall((mid_x, height - 1), (mid_x - 1, height - 1))
            
        maze.dynamic_wall = ((mid_x, height // 2), (mid_x, (height // 2) + 1))

    elif maze_type == "Parallel Paths":
        effective_width = width - 1 
        mid_x = effective_width // 2 
        
        maze.start = (mid_x, 0)
        maze.goal = (mid_x, height - 1)
        start, goal = maze.start, maze.goal

        # Step 1: Generate strict left maze (exactly 1 path inside)
        _generate_dfs(maze, 0, mid_x, 1, height - 2)

        # Step 2: Perfect Mirror left to right
        for y in range(1, height - 2):
            for x in range(mid_x):
                rx = (effective_width - 1) - x
                maze.horizontal_walls[y, rx] = maze.horizontal_walls[y, x]
                maze.horizontal_walls[y+1, rx] = maze.horizontal_walls[y+1, x]
                maze.vertical_walls[y, rx+1] = maze.vertical_walls[y, x]
                maze.vertical_walls[y, rx] = maze.vertical_walls[y, x+1]

        # Step 3: Seal the middle corridor
        for y in range(height):
            maze.vertical_walls[y, mid_x] = True
            maze.vertical_walls[y, mid_x + 1] = True
            maze.vertical_walls[y, effective_width] = True 
            maze.horizontal_walls[y, effective_width] = True
            if y + 1 < height: maze.horizontal_walls[y+1, effective_width] = True

        # Step 4: Connect Start to both halves symmetrically
        maze.remove_wall(start, (mid_x - 1, 0))          
        maze.remove_wall((mid_x - 1, 0), (mid_x - 1, 1)) 
        
        maze.remove_wall(start, (mid_x + 1, 0))          
        maze.remove_wall((mid_x + 1, 0), (mid_x + 1, 1)) 
        
        maze.horizontal_walls[1, mid_x] = True           

        # Step 5: The Detours at the end
        # Left exit (Optimal straight path)
        maze.remove_wall((mid_x - 1, height - 3), (mid_x - 1, height - 2))
        maze.remove_wall((mid_x - 1, height - 2), (mid_x - 1, height - 1))
        maze.remove_wall((mid_x - 1, height - 1), goal)

        # Right exit (Suboptimal detour path)
        maze.remove_wall((mid_x + 1, height - 3), (mid_x + 1, height - 2)) 
        maze.remove_wall((mid_x + 1, height - 2), (mid_x + 2, height - 2)) 
        maze.remove_wall((mid_x + 2, height - 2), (mid_x + 2, height - 1)) 
        maze.remove_wall((mid_x + 2, height - 1), (mid_x + 1, height - 1)) 
        maze.remove_wall((mid_x + 1, height - 1), goal)
        
        maze.horizontal_walls[height - 1, mid_x] = True 

        # Step 6: Bottom random walls to create safe dead-ends
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