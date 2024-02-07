import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Maze dimensions
width, height = 50, 50
cell_size = 16  # Size of each cell in pixels

# Set up the display
window_size = (width * cell_size, height * cell_size)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("50x50 Maze")


def init_maze(width, height):
    # Create a grid filled with walls (1)
    maze = [[1 for _ in range(width)] for _ in range(height)]
    return maze


def carve_maze_from(x, y, maze):
    directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]
    random.shuffle(directions)  # Randomize the carving direction

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 < nx < len(maze[0])-1 and 0 < ny < len(maze)-1 and maze[ny][nx] == 1:
            maze[ny][nx] = 0
            maze[ny-dy//2][nx-dx//2] = 0
            carve_maze_from(nx, ny, maze)


def generate_maze(width, height):
    maze = init_maze(width, height)
    maze[1][1] = 0  # Start carving from the top-left corner
    carve_maze_from(1, 1, maze)
    return maze

# Function to draw the maze


def draw_maze(maze_data):
    for y in range(height):
        for x in range(width):
            rect = pygame.Rect(x*cell_size, y*cell_size, cell_size, cell_size)
            color = (0, 0, 0) if maze_data[y][x] == 1 else (255, 255, 255)
            pygame.draw.rect(screen, color, rect)


# Main game loop
running = True
maze_data = generate_maze(width, height)  # Generate the maze data
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # Fill the screen with white
    draw_maze(maze_data)  # Draw the maze

    pygame.display.flip()  # Update the display

pygame.quit()
sys.exit()
