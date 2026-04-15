import numpy as np
import matplotlib.pyplot as plt
from math import log2

class MazeEnvironment:
    """2D-grid"""
    def __init__(self, width, height, start, goal, walls=None):
        self.width = width
        self.height = height
        self.start = start  # (x, y)
        self.goal = goal    # (x, y)
        self.grid = np.zeros((height, width))
        if walls:
            for (x, y) in walls:
                if 0 <= x < width and 0 <= y < height:
                    self.grid[y, x] = 1 # 1 is een muur

    def is_valid(self, x, y):
        """Controleert of een cel binnen het grid valt en geen muur is."""
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] == 0

def visualize_maze(maze, title="Maze Visualisatie"):
    """
    Maakt een grafische weergave van het doolhof.
    Zwart = Muur, Wit = Pad, S = Start, G = Goal.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Teken het grid (cmap 'Greys' maakt 1 zwart en 0 wit)
    ax.imshow(maze.grid, cmap='Greys', origin='upper')
    
    # Voeg letters toe voor Start en Goal
    # Gebruik maze.start[0] voor X (kolom) en [1] voor Y (rij)
    ax.text(maze.start[0], maze.start[1], 'S', ha='center', va='center', 
            color='green', weight='bold', fontsize=20)
    ax.text(maze.goal[0], maze.goal[1], 'G', ha='center', va='center', 
            color='red', weight='bold', fontsize=20)
    
    # Teken de grid-lijnen precies tussen de cellen
    ax.set_xticks(np.arange(-.5, maze.width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, maze.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Verwijder de standaard as-nummers voor een schoner uiterlijk
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_title(title)
    plt.show()

def calculate_shannon_entropy(population_positions):
    """Berekent diversiteit op basis van de locaties van agents."""
    if not population_positions:
        return 0
    counts = {}
    for pos in population_positions:
        counts[pos] = counts.get(pos, 0) + 1
    
    n = len(population_positions)
    entropy = 0
    for count in counts.values():
        p = count / n
        entropy -= p * log2(p)
    return entropy

class GeneticAlgorithm:
    def __init__(self, maze, pop_size=50):
        self.maze = maze
        self.pop_size = pop_size

class PSO:
    def __init__(self, maze, num_particles=30):
        self.maze = maze

class ACO:
    def __init__(self, maze):
        self.maze = maze
        self.pheromones = np.ones((maze.height, maze.width)) * 0.1


if __name__ == "__main__":
    simple_walls = [
        (1, 1), (1, 2), (1, 3), 
        (3, 0), (3, 1), (3, 2),
        (0, 3), (2, 4)
    ]
    
    maze = MazeEnvironment(width=5, height=5, start=(0, 0), goal=(4, 4), walls=simple_walls)
    
    visualize_maze(maze, title="Simple Maze Setup (Groep 27)")