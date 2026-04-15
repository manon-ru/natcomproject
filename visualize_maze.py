import numpy as np
import matplotlib.pyplot as plt
import random

class MazeEnvironment:
    """
    Doolhof omgeving waarbij muren de doorgang tussen cellen blokkeren.
    """
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        # Muren als booleans (True = muur aanwezig)
        self.horizontal_walls = np.ones((height + 1, width), dtype=bool)
        self.vertical_walls = np.ones((height, width + 1), dtype=bool)

    def remove_wall(self, c1, c2):
        x1, y1 = c1
        x2, y2 = c2
        if x1 == x2: # Verticaal verplaatsen -> horizontale muur weghalen
            self.horizontal_walls[max(y1, y2), x1] = False
        elif y1 == y2: # Horizontaal verplaatsen -> verticale muur weghalen
            self.vertical_walls[y1, max(x1, x2)] = False

def generate_deterministic_maze(width, height, seed_value=42):
    """
    Genereert een doolhof met een vaste seed voor reproduceerbaarheid.
    """
    # Zet de seed vast voor zowel random als numpy
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    maze = MazeEnvironment(width, height, (0, 0), (width-1, height-1))
    visited = set([(0, 0)])
    stack = [(0, 0)]

    while stack:
        x, y = stack[-1]
        neighbors = []
        # Check mogelijke buren
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                neighbors.append((nx, ny))
        
        if neighbors:
            # Kies een buur op basis van de seed
            next_cell = neighbors[random.randint(0, len(neighbors) - 1)]
            maze.remove_wall((x, y), next_cell)
            visited.add(next_cell)
            stack.append(next_cell)
        else:
            stack.pop()
    return maze

def visualize_complex_maze(maze, title="Project Group 27: Consistent Maze"):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Teken de horizontale muren
    for y in range(maze.height + 1):
        for x in range(maze.width):
            if maze.horizontal_walls[y, x]:
                ax.plot([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], color='black', lw=2)
                
    # Teken de verticale muren
    for y in range(maze.height):
        for x in range(maze.width + 1):
            if maze.vertical_walls[y, x]:
                ax.plot([x - 0.5, x - 0.5], [y - 0.5, y + 0.5], color='black', lw=2)

    # Start (S) en Goal (G)
    ax.text(maze.start[0], maze.start[1], 'S', color='green', weight='bold', 
            ha='center', va='center', fontsize=15)
    ax.text(maze.goal[0], maze.goal[1], 'G', color='red', weight='bold', 
            ha='center', va='center', fontsize=15)

    ax.set_aspect('equal')
    ax.set_ylim(maze.height - 0.5, -0.5)
    ax.axis('off')
    plt.title(title)
    plt.show()

# --- Uitvoering ---
if __name__ == "__main__":
    # Door deze seed te gebruiken, krijg je altijd hetzelfde doolhof
    # Ideaal voor de 'efficiency tests' uit jullie voorstel.
    MY_SEED = 2026 
    
    # Maak een maze van medium complexiteit (Fase 2 van jullie plan)
    maze = generate_deterministic_maze(10, 10, seed_value=MY_SEED)
    visualize_complex_maze(maze, title=f"Maze met Seed: {MY_SEED}")