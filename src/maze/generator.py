import random
from turtle import width
import numpy as np
from maze.environment import MazeEnvironment

braid_factor = 0.3  # Adjust this to control how many extra paths are added (0.0 = perfect maze, 1.0 = very braided)

def _braid_maze(maze, x1, x2, y1, y2, factor):
    """
    Randomly removes internal walls within a region to create multiple paths.
    factor: 0.0 (perfect maze) to 1.0 (very sparse/open).
    """
    if factor <= 0: return
    
    for y in range(y1, y2):
        for x in range(x1, x2):
            # Randomly check horizontal and vertical walls
            for dx, dy in [(1, 0), (0, 1)]:
                nx, ny = x + dx, y + dy
                if x1 <= nx < x2 and y1 <= ny < y2:
                    if maze.has_wall_between((x, y), (nx, ny)):
                        if random.random() < factor:
                            maze.remove_wall((x, y), (nx, ny))


def _generate_dfs(maze, x1, x2, y1, y2):
    # Recursive DFS-based maze generator for creating a perfect maze in the specified region.
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
    
    # Define start and goal positions based on maze type
    if maze_type in ["Parallel Paths", "Sudden Wall"]:
        start, goal = (width // 2, 0), (width // 2, height - 1)
    elif maze_type == "U-Trap":
        start, goal = (0, height // 2), (width - 1, height // 2)
    else:
        start, goal = (0, 0), (width - 1, height - 1)

    maze = MazeEnvironment(width, height, start=start, goal=goal)

    if maze_type == "Random":
        _generate_dfs(maze, 0, width, 0, height)
        # Post-process to add multiple paths
        _braid_maze(maze, 0, width, 0, height, braid_factor)
        
    elif maze_type == "U-Trap":
        mid_y = height // 2
        # Generate and then braid each half independently to maintain the trap's logic
        _generate_dfs(maze, 0, width, 0, mid_y)
        _braid_maze(maze, 0, width, 0, mid_y, braid_factor)
        
        _generate_dfs(maze, 0, width, mid_y, height)
        _braid_maze(maze, 0, width, mid_y, height, braid_factor)

        # Step 2: Hermetically seal the border between the two halves
        for x in range(width):
            maze.horizontal_walls[mid_y, x] = True

        # Step 3: Connect Start (0, mid_y) to the True Path (top half)
        # The agent must go UP to actually solve the maze.
        maze.remove_wall((0, mid_y), (0, mid_y - 1))

        # Step 4: Isolate the Goal (width - 1, mid_y) from the Trap
        # We build solid walls on the left and bottom of the goal cell.
        maze.add_wall((width - 1, mid_y), (width - 2, mid_y))
        if mid_y + 1 < height:
            maze.add_wall((width - 1, mid_y), (width - 1, mid_y + 1))
        
        # Step 5: Connect Goal to the True Path (top half)
        # The agent must drop down from the top half to finish.
        maze.remove_wall((width - 1, mid_y), (width - 1, mid_y - 1))

        # Step 6: Create the ultimate deceptive corridor
        # We clear a straight line from Start directly toward the Goal.
        # This forces the heuristic distance to decrease smoothly to 1,
        # but the agent will hit the wall we built in Step 4.
        for x in range(width - 2):
            maze.remove_wall((x, mid_y), (x + 1, mid_y))

    elif maze_type == "Sudden Wall":
        mid_x = width // 2
        # Generate and braid the wings
        _generate_dfs(maze, 0, mid_x, 0, height)
        _braid_maze(maze, 0, mid_x, 0, height, braid_factor)
        
        _generate_dfs(maze, mid_x + 1, width, 0, height)
        _braid_maze(maze, mid_x + 1, width, 0, height, braid_factor)
        
        # Step 2: Carve the Short Path (The Central Highway)
        for y in range(height - 1):
            maze.remove_wall((mid_x, y), (mid_x, y + 1))
            # Seal the highway walls so agents stay in the tube initially
            if 0 < y < height - 1:
                maze.vertical_walls[y, mid_x] = True
                maze.vertical_walls[y, mid_x + 1] = True
            
        # Step 3: Open "Valves" to both wings
        # We connect the start/goal area to BOTH the left and right sectors
        for side_x in [mid_x - 1, mid_x + 1]:
            # Connect top (near Start)
            maze.remove_wall((mid_x, 0), (side_x, 0))
            # Connect bottom (near Goal)
            maze.remove_wall((mid_x, height - 1), (side_x, height - 1))
            
        # Step 4: Define the Disruption (Exactly in the middle of the highway)
        maze.dynamic_wall = ((mid_x, height // 2), (mid_x, (height // 2) + 1))

    elif maze_type == "Parallel Paths":
        # Maze 3: Parallel Paths + small detour at the end
        mid_x = width // 2
        _generate_dfs(maze, 0, mid_x, 1, height - 2)
        _braid_maze(maze, 0, mid_x, 1, height - 2, braid_factor)

        # Step 2: Mirror walls to the right half
        for y in range(1, height - 2):
            for x in range(mid_x):
                rx = width - 1 - x
                maze.horizontal_walls[y, rx] = maze.horizontal_walls[y, x]
                maze.horizontal_walls[y+1, rx] = maze.horizontal_walls[y+1, x]
                maze.vertical_walls[y, rx+1] = maze.vertical_walls[y, x]
                maze.vertical_walls[y, rx] = maze.vertical_walls[y, x+1]

        # Step 3: close the middle corridor to create two parallel paths
        for y in range(1, height - 1):
            maze.vertical_walls[y, mid_x] = True
            maze.vertical_walls[y, mid_x + 1] = True # Zorg dat de mid-kolom echt dicht is

        # Step 4: Connect Start to both halves (Entrance)
        maze.remove_wall(start, (mid_x - 1, 0)) # Door to the left
        maze.remove_wall(start, (mid_x + 1, 0)) # Door to the right
        maze.remove_wall((mid_x - 1, 0), (mid_x - 1, 1)) # Corridor to the left maze
        maze.remove_wall((mid_x + 1, 0), (mid_x + 1, 1)) # Corridor to the right maze

        # Step 5: THE GEARDED DETOUR AT THE END
        # Left exit: Straight and efficient (3 steps to the goal)
        maze.remove_wall((mid_x - 1, height - 3), (mid_x - 1, height - 2))
        maze.remove_wall((mid_x - 1, height - 2), (mid_x - 1, height - 1))
        maze.remove_wall((mid_x - 1, height - 1), goal)

        # Right exit: The small detour! (Exactly 5 steps to the goal)
        maze.remove_wall((mid_x + 1, height - 3), (mid_x + 1, height - 2))
        maze.remove_wall((mid_x + 1, height - 2), (mid_x + 2, height - 2))
        maze.remove_wall((mid_x + 2, height - 2), (mid_x + 2, height - 1))
        maze.remove_wall((mid_x + 2, height - 1), (mid_x + 1, height - 1))
        maze.remove_wall((mid_x + 1, height - 1), goal)
        
        # Close the middle corridor all the way down
        maze.horizontal_walls[height - 1, mid_x + 1] = True 

        # Step 6: Add some random walls in the bottom row to create dead-ends and force decision-making
        for _ in range(4): # Add 4 random walls in the bottom row
            for y in range(height - 2, height):
                for x in range(width):
                    # We only add walls in the bottom row, and we avoid blocking the direct paths to the goal
                    if maze.horizontal_walls[y, x] and maze.horizontal_walls[y+1, x] and maze.vertical_walls[y, x] and maze.vertical_walls[y, x+1]:
                        neighbors = []
                        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            nx, ny = x + dx, y + dy
                            # Check if the neighbor is within bounds and not on the middle corridor
                            if maze.in_bounds(nx, ny) and not (x < mid_x and nx >= mid_x) and not (x > mid_x and nx <= mid_x):
                                # Check if there's currently a wall between (x, y) and (nx, ny)
                                if not (maze.horizontal_walls[ny, nx] and maze.horizontal_walls[ny+1, nx] and maze.vertical_walls[ny, nx] and maze.vertical_walls[ny, nx+1]):
                                    # We only want to add a wall if it doesn't block the direct path to the goal
                                    if (nx, ny) != goal:
                                        neighbors.append((nx, ny))
                        
                        if neighbors:
                            maze.remove_wall((x, y), random.choice(neighbors))

    return maze