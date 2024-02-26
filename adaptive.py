import pygame
from random import choice
import os
from pygame.locals import *
from queue import PriorityQueue
import heapq
import random
import string
import copy


class Cell(pygame.sprite.Sprite):
    w, h = 8, 8

    def __init__(self, x, y, maze):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([self.w, self.h])
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.x = x * self.w
        self.rect.y = y * self.h
        self.h = float('inf')  # allow cell to store h value

        self.x = x
        self.y = y
        self.maze = maze
        self.nbs = [(x + nx, y + ny) for nx, ny in ((-2, 0), (0, -2), (2, 0), (0, 2))
                    if 0 <= x + nx < maze.w and 0 <= y + ny < maze.h]
        self.name = ''.join(random.choice(string.ascii_letters)
                            for i in range(10))

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def update_heuristic(self, g_cost):
        # print("in here")
        self.h = g_cost


class Wall(Cell):
    def __init__(self, x, y, maze):
        super(Wall, self).__init__(x, y, maze)
        self.image.fill((0, 0, 0))
        self.type = 0


class Fog(Cell):
    def __init__(self, x, y, maze):
        super(Fog, self).__init__(x, y, maze)
        self.image.fill((100, 100, 100))
        self.type = 1


class Maze:
    def __init__(self, size):
        self.w, self.h = size[0] // Cell.w, size[1] // Cell.h
        self.grid = [[Wall(x, y, self) for y in range(self.h)]
                     for x in range(self.w)]

    def get(self, x, y) -> Cell:
        return self.grid[x][y]

    def place_wall(self, x, y):
        self.grid[x][y] = Wall(x, y, self)

    def draw(self, screen):
        for row in self.grid:
            for cell in row:
                cell.draw(screen)

    def generate(self, screen=None, animate=False):
        unvisited = [c for r in self.grid for c in r if c.x % 2 and c.y % 2]
        cur = unvisited.pop()
        stack = []

        while unvisited:
            try:
                n = choice(
                    [c for c in map(lambda x: self.get(*x), cur.nbs) if c in unvisited])
                stack.append(cur)
                nx, ny = cur.x - (cur.x - n.x) // 2, cur.y - (cur.y - n.y) // 2
                self.grid[nx][ny] = Cell(nx, ny, self)
                self.grid[cur.x][cur.y] = Cell(cur.x, cur.y, self)
                cur = n
                unvisited.remove(n)

                if animate:
                    self.draw(screen)
                    pygame.display.update()
                    pygame.time.wait(10)
            except IndexError:
                if stack:
                    cur = stack.pop()

    def neighbors(self, cell):  # returns a list of Cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nbs = []
        for dx, dy in directions:
            nx, ny = cell.x + dx, cell.y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and not isinstance(self.get(nx, ny), Wall):
                nbs.append(self.get(nx, ny))
        return nbs


# Manhattan distance
def heuristic(a: Cell, b: Cell) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def adaptive_a_star_search_experiment(maze, start, goal, use_updated_heuristic=True):
    # I need to make sure that the algorithm reruns to show that its choosing more efficient paths
    open_set = PriorityQueue()
    start_coords = (start.x, start.y)
    goal_coords = (goal.x, goal.y)
    # heap has structure of (priority, coords)
    open_set.put((0, start_coords))
    came_from = {}
    # I need to create an explicit closed list.
    # rn I'm using cost_so_far to retrace my steps
    g_cost = {}
    # came_from is a dictionary that maps a node to the parent node
    came_from[start_coords] = None
    g_cost[start_coords] = 0

    expanded_nodes = 0

    while not open_set.empty():
        current_priority, current_coords = open_set.get()
        # Increment counter each time a node is expanded
        expanded_nodes += 1
        # unpacking the tuple current_coords into maze
        current = maze.get(*current_coords)
        if current_coords == goal_coords:
            # print("Goal reached!")
            # print(g_cost)
            if not use_updated_heuristic:
                print("in here")
                update_all_heuristics(came_from, g_cost, goal, maze)
            break

        """
        1. Iterate through the neighbors of the current node.
        2. For each neighbor, calculate its coordinates.
        3. Determine the new path cost to reach this neighbor (current cost to reach the current node + 1)
        4. if the neighbor has not been visited or the new cost is lower, update the open list to include the neighbor
        """
        for next in maze.neighbors(current):
            next_coords = (next.x, next.y)
            new_cost = g_cost[current_coords] + 1
            if next_coords not in g_cost or new_cost < g_cost[next_coords]:
                g_cost[next_coords] = new_cost
                priority = new_cost + \
                    (next.h if use_updated_heuristic else heuristic(goal, next))
                next.update_heuristic(heuristic(goal, next))
                open_set.put((priority, next_coords))
                came_from[next_coords] = current_coords

    return expanded_nodes


def adaptive_a_star_search(maze, start, goal, use_updated_heuristic=False):
    # I need to make sure that the algorithm reruns to show that its choosing more efficient paths
    open_set = PriorityQueue()
    start_coords = (start.x, start.y)
    goal_coords = (goal.x, goal.y)
    # heap has structure of (priority, coords)
    open_set.put((0, start_coords))
    came_from = {}
    # I need to create an explicit closed list.
    # rn I'm using cost_so_far to retrace my steps
    g_cost = {}
    # came_from is a dictionary that maps a node to the parent node
    came_from[start_coords] = None
    g_cost[start_coords] = 0

    while not open_set.empty():
        current_priority, current_coords = open_set.get()
        # unpacking the tuple current_coords into maze
        current = maze.get(*current_coords)
        if current_coords == goal_coords:
            # print("Goal reached!")
            # print(g_cost)
            if not use_updated_heuristic:
                print("in here")
                update_all_heuristics(came_from, g_cost, goal, maze)
            break

        """
        1. Iterate through the neighbors of the current node.
        2. For each neighbor, calculate its coordinates.
        3. Determine the new path cost to reach this neighbor (current cost to reach the current node + 1)
        4. if the neighbor has not been visited or the new cost is lower, update the open list to include the neighbor
        """
        for next in maze.neighbors(current):
            next_coords = (next.x, next.y)
            new_cost = g_cost[current_coords] + 1
            if next_coords not in g_cost or new_cost < g_cost[next_coords]:
                g_cost[next_coords] = new_cost
                priority = new_cost + \
                    (next.h if use_updated_heuristic else heuristic(goal, next))
                next.update_heuristic(heuristic(goal, next))
                open_set.put((priority, next_coords))
                came_from[next_coords] = current_coords

    return came_from


def update_all_heuristics(came_from, g_cost, goal, maze):
    g_goal = g_cost[(goal.x, goal.y)]
    print("g_goal: ", g_goal)
    for coords, _ in came_from.items():
        node = maze.get(*coords)
        # print("node: ", node.h)
        # Update heuristic based on most recent search
        print("node.h value before update: ", node.h)
        node.h = g_goal - g_cost[coords]
        print("node.h value after update: ", node.h)


def reconstruct_path(came_from, start, goal, maze):
    current = (goal.x, goal.y)
    start_coords = (start.x, start.y)
    path = []
    while current != start_coords:
        x, y = current
        path.append(maze.get(x, y))
        current = came_from.get(current)
        if current is None:
            break
    path.append(start)
    path.reverse()
    return path


# Default color is green, full opacity
def draw_path(screen, path, color=(0, 255, 0), opacity=255):
    # Create a temporary surface with per-pixel alpha
    temp_surface = pygame.Surface((Cell.w, Cell.h), pygame.SRCALPHA)
    # Set the color of the temporary surface, including the alpha for opacity
    # Add the opacity value to the color tuple
    semi_transparent_color = color + (opacity,)
    temp_surface.fill(semi_transparent_color)

    for node in path:
        # Blit the temporary surface onto the screen at the node's position
        screen.blit(temp_surface, (node.rect.x, node.rect.y))


def adaptive_a_star_maze_experiment(screen, num_mazes):
    expanded_cells = []
    for i in range(int(num_mazes)):
        print("Testing maze", i+1, "out of", num_mazes, "!")
        maze = Maze(WINSIZE)
        maze.generate(screen, True)
        start = maze.get(1, 1)
        goal = maze.get(maze.w - 2, maze.h - 2)
        # first search
        adaptive_a_star_search(maze, start, goal)
        # second search
        expanded_nodes = adaptive_a_star_search_experiment(
            maze, start, goal)

        expanded_cells.append(expanded_nodes)

    print("sum of expanded cells", sum(expanded_cells))


def draw_maze(screen):
    maze = Maze(WINSIZE)
    maze.generate(screen, True)
    start = maze.get(1, 1)
    goal = maze.get(maze.w - 2, maze.h - 2)

    came_from = adaptive_a_star_search(maze, start, goal)
    path = reconstruct_path(came_from, start, goal, maze)
    draw_path(screen, path, color=(0, 255, 0))

    # second run through
    came_from_updated = adaptive_a_star_search(
        maze, start, goal, use_updated_heuristic=True)
    path_updated = reconstruct_path(came_from_updated, start, goal, maze)
    draw_path(screen, path_updated, color=(255, 0, 0), opacity=60)


"""Winsize sets the dimension of the maze. Make sure it's an odd number. """
WINSIZE = (Cell.w * 103, Cell.h * 103)


def main():
    pygame.init()
    scr_inf = pygame.display.Info()
    os.environ['SDL_VIDEO_WINDOW_POS'] = '{}, {}'.format(scr_inf.current_w // 2 - WINSIZE[0] // 2,
                                                         scr_inf.current_h // 2 - WINSIZE[1] // 2)
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('Maze with A* Pathfinding')
    screen.fill((0, 0, 0))

    clock = pygame.time.Clock()
    # draw_maze(screen)

    num_mazes = input("How many mazes would you like to generate? ")
    # print("OK! Generating", command, "mazes...")
    # maze_list = generate_maze_list(screen, command)
    # print("Finished!")
    command = input("What would you like to do? ")
    if command == "1":
        adaptive_a_star_maze_experiment(screen, num_mazes)
    else:
        done = False
        while not done:
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    done = True

            pygame.display.update()
            clock.tick(60)


if __name__ == '__main__':
    main()
