import random
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
        # MAZE 1: U-Trap
        mid_y = height // 2
        
        # Stap 1: Bovenste helft wordt de extreem complexe, lange omweg
        _generate_dfs(maze, 0, width, 0, mid_y)
        
        # Stap 2: Onderste helft wordt het misleidende doolhof richting de val
        _generate_dfs(maze, 0, width - 2, mid_y, height)

        # Stap 3: Bouw de letterlijke 3x3 cirkel (de val) aan de rechterkant
        cx, cy = width - 4, mid_y + 1
        ring = [(cx-1, cy-1), (cx, cy-1), (cx+1, cy-1),
                (cx+1, cy), (cx+1, cy+1), (cx, cy+1),
                (cx-1, cy+1), (cx-1, cy)]
        for i in range(8):
            maze.remove_wall(ring[i], ring[(i+1)%8])

        # Stap 4: Sluit de Goal hermetisch af van de onderste helft!
        # Mieren MOGEN de goal niet bereiken vanuit de val.
        maze.vertical_walls[mid_y, width - 2] = True
        maze.vertical_walls[mid_y+1, width - 2] = True
        maze.horizontal_walls[mid_y, width - 1] = True

        # Stap 5: Verbind Start met beide werelden
        maze.remove_wall(start, (0, mid_y - 1)) # Naar de lange omweg
        maze.remove_wall(start, (0, mid_y + 1)) # Naar de val/cirkel

        # Stap 6: Verbind de lange omweg (bovenlangs) met de Goal
        maze.remove_wall((width - 1, mid_y - 1), goal)

    elif maze_type == "Sudden Wall":
        # MAZE 2: KORT VS LANG (Beide zijn nu 100% complexe doolhoven!)
        mid_x = width // 2
        
        # Sector 1 (Kort): Een smalle, maar extreem kronkelige corridor in het midden
        _generate_dfs(maze, mid_x - 2, mid_x + 2, 1, height - 1)
        
        # Sector 2 (Lang): Een gigantisch, kronkelig doolhof op rechts
        _generate_dfs(maze, mid_x + 2, width, 1, height - 1)
        
        # (Visuele opvulling voor links, zodat het niet leeg is)
        _generate_dfs(maze, 0, mid_x - 2, 1, height - 1)

        # Isoleer de gebieden luchtdicht van elkaar
        for y in range(height):
            maze.vertical_walls[y, mid_x + 2] = True
            maze.vertical_walls[y, mid_x - 2] = True

        # Verbind Start/Goal met het korte pad
        maze.remove_wall(start, (mid_x, 1))
        maze.remove_wall(goal, (mid_x, height - 2))

        # Verbind Start/Goal met het lange pad via de randen
        for x in range(mid_x, mid_x + 3):
            maze.remove_wall((x, 0), (x+1, 0))
            maze.remove_wall((x, height-1), (x+1, height-1))
        maze.remove_wall((mid_x + 3, 0), (mid_x + 3, 1))
        maze.remove_wall((mid_x + 3, height - 1), (mid_x + 3, height - 2))

    elif maze_type == "Parallel Paths":
        # MAZE 3: 100% GESPIEGELD + EEN GEGARANDEERDE DETOUR
        mid_x = width // 2
        
        # Stap 1: Vul de linkerhelft met een zwaar doolhof
        _generate_dfs(maze, 0, mid_x, 1, height - 1)

        # Stap 2: Spiegel MUUR VOOR MUUR naar de rechterhelft
        for y in range(1, height - 1):
            for x in range(mid_x):
                rx = width - 1 - x
                maze.horizontal_walls[y, rx] = maze.horizontal_walls[y, x]
                maze.horizontal_walls[y+1, rx] = maze.horizontal_walls[y+1, x]
                maze.vertical_walls[y, rx+1] = maze.vertical_walls[y, x]
                maze.vertical_walls[y, rx] = maze.vertical_walls[y, x+1]

        # Stap 3: Scheid de helften luchtdicht in het midden, 
        # MAAR laat y=0 (start) en y=height-1 (goal) met rust!
        for y in range(1, height - 1):
            maze.vertical_walls[y, mid_x] = True
            maze.vertical_walls[y, mid_x + 1] = True # Zorg dat de mid-kolom echt dicht is

        # Stap 4: Verbind Start en Goal aan beide helften
        # (Dit doen we PAS HIERNA, zodat we onze eigen deuren niet meer dichtgooien)
        maze.remove_wall(start, (mid_x - 1, 0)) # Deur naar links
        maze.remove_wall(start, (mid_x + 1, 0)) # Deur naar rechts
        maze.remove_wall((mid_x - 1, 0), (mid_x - 1, 1)) # Gangetje naar linker doolhof
        maze.remove_wall((mid_x + 1, 0), (mid_x + 1, 1)) # Gangetje naar rechter doolhof

        maze.remove_wall(goal, (mid_x - 1, height - 1))
        maze.remove_wall(goal, (mid_x + 1, height - 1))
        maze.remove_wall((mid_x - 1, height - 1), (mid_x - 1, height - 2))
        maze.remove_wall((mid_x + 1, height - 1), (mid_x + 1, height - 2))

        # Stap 5: DE GEGARANDEERDE DETOUR
        # We gebruiken BFS om de exacte route op rechts te vinden, en breken hem af!
        path = _find_shortest_path(maze, (mid_x + 1, 1), (mid_x + 1, height - 2))
        if path and len(path) > 4:
            # Pak een coördinaat precies in het midden van de route
            idx = len(path) // 2
            px, py = path[idx]
            nx, ny = path[idx + 1]

            # Zet de muur TERUG op de hoofdroute om hem te blokkeren
            if px == nx: maze.horizontal_walls[max(py, ny), px] = True
            if py == ny: maze.vertical_walls[py, max(px, nx)] = True

            # Graaf een dwingende u-bocht om de nieuwe muur heen
            if px == nx: # Route ging verticaal
                dx = 1 if px < width - 1 else -1
                maze.remove_wall((px, py), (px + dx, py))
                maze.remove_wall((px + dx, py), (px + dx, ny))
                maze.remove_wall((px + dx, ny), (nx, ny))
            else: # Route ging horizontaal
                dy = 1 if py < height - 1 else -1
                maze.remove_wall((px, py), (px, py + dy))
                maze.remove_wall((px, py + dy), (nx, py + dy))
                maze.remove_wall((nx, py + dy), (nx, ny))

    return maze

# --- KERN FUNCTIES ---

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