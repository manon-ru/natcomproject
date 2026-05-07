import random
from turtle import width
import numpy as np
from maze.environment import MazeEnvironment

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
        
    elif maze_type == "U-Trap":
        # MAZE 1: U-Trap (Deception)
        mid_y = height // 2
        
        # Step 1: Generate two completely isolated halves
        # Top half (The True Path)
        _generate_dfs(maze, 0, width, 0, mid_y)
        # Bottom half (The Trap)
        _generate_dfs(maze, 0, width, mid_y, height)

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
        # MAZE 2: Sudden Wall (Dynamic Adaptability)
        mid_x = width // 2
        start = (mid_x, 0)
        goal = (mid_x, height - 1)
        maze.start = start
        maze.goal = goal
        
        # Step 1: Generate DFS background noise
        _generate_dfs(maze, 0, width, 0, height)
        
        # Step 2: Carve the Short Path (Straight down the middle)
        for y in range(height - 1):
            maze.remove_wall((mid_x, y), (mid_x, y + 1))
            # Seal the left and right sides so agents don't wander off the highway
            if y > 0 and y < height - 1:
                maze.vertical_walls[y, mid_x] = True
                maze.vertical_walls[y, mid_x + 1] = True
            
        # Step 3: Carve the Long Path (Around the left perimeter)
        # Connect start to the far left wall
        for x in range(1, mid_x + 1):
            maze.remove_wall((x, 0), (x - 1, 0))
            maze.remove_wall((x, height - 1), (x - 1, height - 1))
        # Clear the far left column
        for y in range(height - 1):
            maze.remove_wall((0, y), (0, y + 1))
            
        # Step 4: Define the Disruption
        # This specifies the exact wall that will be ADDED during the simulation
        maze.dynamic_wall = ((mid_x, height // 2), (mid_x, (height // 2) + 1))

    elif maze_type == "Parallel Paths":
        # MAZE 3: 100% GESPIEGELD + EEN GEGARANDEERDE DETOUR AAN HET EIND
        mid_x = width // 2

        # Stap 1: Vul de linkerhelft met een zwaar doolhof
        # We stoppen 1 rij eerder (height - 2) zodat we ruimte hebben voor de detour
        _generate_dfs(maze, 0, mid_x, 1, height - 2)

        # Stap 2: Spiegel MUUR VOOR MUUR naar de rechterhelft
        for y in range(1, height - 2):
            for x in range(mid_x):
                rx = width - 1 - x
                maze.horizontal_walls[y, rx] = maze.horizontal_walls[y, x]
                maze.horizontal_walls[y+1, rx] = maze.horizontal_walls[y+1, x]
                maze.vertical_walls[y, rx+1] = maze.vertical_walls[y, x]
                maze.vertical_walls[y, rx] = maze.vertical_walls[y, x+1]

        # Stap 3: Scheid de helften luchtdicht in het midden
        for y in range(1, height - 1):
            maze.vertical_walls[y, mid_x] = True
            maze.vertical_walls[y, mid_x + 1] = True # Zorg dat de mid-kolom echt dicht is

        # Stap 4: Verbind Start met beide helften (Ingang)
        maze.remove_wall(start, (mid_x - 1, 0)) # Deur naar links
        maze.remove_wall(start, (mid_x + 1, 0)) # Deur naar rechts
        maze.remove_wall((mid_x - 1, 0), (mid_x - 1, 1)) # Gangetje naar linker doolhof
        maze.remove_wall((mid_x + 1, 0), (mid_x + 1, 1)) # Gangetje naar rechter doolhof

        # Stap 5: DE GEGARANDEERDE DETOUR AAN HET EIND
        # Linker uitgang: Kaarsrecht en efficiënt (3 stappen naar de goal)
        maze.remove_wall((mid_x - 1, height - 3), (mid_x - 1, height - 2))
        maze.remove_wall((mid_x - 1, height - 2), (mid_x - 1, height - 1))
        maze.remove_wall((mid_x - 1, height - 1), goal)

        # Rechter uitgang: De kleine detour! (Exact 5 stappen naar de goal)
        maze.remove_wall((mid_x + 1, height - 3), (mid_x + 1, height - 2))
        
        # Slinger naar rechts: (mid_x+1) -> (mid_x+2) -> omlaag -> (mid_x+1) -> goal
        maze.remove_wall((mid_x + 1, height - 2), (mid_x + 2, height - 2))
        maze.remove_wall((mid_x + 2, height - 2), (mid_x + 2, height - 1))
        maze.remove_wall((mid_x + 2, height - 1), (mid_x + 1, height - 1))
        maze.remove_wall((mid_x + 1, height - 1), goal)
        
        # Sluit het directe pad op de rechterhelft hermetisch af
        maze.horizontal_walls[height - 1, mid_x + 1] = True 

        # Stap 6: Organische opvulling van de lege ruimte (Voorkomt "losse blokjes")
        for _ in range(4): # Loop een paar keer zodat alle blokjes met de route versmelten
            for y in range(height - 2, height):
                for x in range(width):
                    # Als de cel nog 4 muren heeft (compleet dicht blokje)
                    if maze.horizontal_walls[y, x] and maze.horizontal_walls[y+1, x] and maze.vertical_walls[y, x] and maze.vertical_walls[y, x+1]:
                        neighbors = []
                        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            nx, ny = x + dx, y + dy
                            # Blijf binnen de grenzen én steek de middellijn niet over
                            if maze.in_bounds(nx, ny) and not (x < mid_x and nx >= mid_x) and not (x > mid_x and nx <= mid_x):
                                # Check of de buurcel AL open is (deel van het doolhof)
                                if not (maze.horizontal_walls[ny, nx] and maze.horizontal_walls[ny+1, nx] and maze.vertical_walls[ny, nx] and maze.vertical_walls[ny, nx+1]):
                                    # Voorkom dat we per ongeluk direct in de Goal breken
                                    if (nx, ny) != goal:
                                        neighbors.append((nx, ny))
                        
                        if neighbors:
                            maze.remove_wall((x, y), random.choice(neighbors))

    return maze


def _generate_dfs(maze, x1, x2, y1, y2):
    """Genereert een perfect doolhof (honderden kronkels, GEEN rechte lijnen)."""
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

def _has_wall(maze, c1, c2):
    """Checkt of er een muur tussen twee cellen staat."""
    x1, y1 = c1
    x2, y2 = c2
    if x1 == x2: return maze.horizontal_walls[max(y1, y2), x1]
    if y1 == y2: return maze.vertical_walls[y1, max(x1, x2)]
    return True

def _find_shortest_path(maze, start, goal):
    """Interne BFS om tijdens de generatie de route te vinden en te manipuleren."""
    queue = [[start]]
    visited = {start}
    while queue:
        path = queue.pop(0)
        curr = path[-1]
        if curr == goal: return path
        x, y = curr
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if maze.in_bounds(nx, ny) and (nx, ny) not in visited:
                if not _has_wall(maze, curr, (nx, ny)):
                    visited.add((nx, ny))
                    queue.append(path + [(nx, ny)])
    return []