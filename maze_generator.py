import pygame
from random import choice
import os
from pygame.locals import *
from queue import PriorityQueue
import heapq
import random
import string
import pickle


class Cell(pygame.sprite.Sprite):
    w, h = 8, 8 # determines size of cells

    def __init__(self, x, y, maze):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([self.w, self.h])
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.x = x * self.w
        self.rect.y = y * self.h

        self.x = x
        self.y = y
        self.maze = maze
        self.nbs = [(x + nx, y + ny) for nx, ny in ((-2, 0), (0, -2), (2, 0), (0, 2))
                    if 0 <= x + nx < maze.w and 0 <= y + ny < maze.h]
        self.name = ''.join(random.choice(string.ascii_letters) for i in range(10))

    def draw(self, screen):
        screen.blit(self.image, self.rect)


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
                    #pygame.time.wait(10)
            except IndexError:
                if stack:
                    cur = stack.pop()
        if isinstance(self.grid[1][1], Wall): 
            self.grid[1][1] = Cell(1,1,self)
            #print("found a wall cell at (1,1)!")

    def neighbors(self, cell): # returns a list of Cells
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


# For each neighbor of the current node:
# Calculate its coordinates.
# Determine the new path cost to reach this neighbor (current cost to reach the current node + 1)
# If the neighbor has not been visited or the new cost is lower than a previously recorded cost, update cost_so_far and came_from for the neighbor.
def a_star_search(maze, start, goal):
    frontier = PriorityQueue()
    start_coords = (start.x, start.y)
    goal_coords = (goal.x, goal.y)
    # heap has structure of (priority, coords)
    frontier.put((0, start_coords))
    came_from = {}
    # I need to create an explicit closed list.
    # rn I'm using cost_so_far to retrace my steps
    cost_so_far = {}
    # came_from is a dictionary that maps a node to the parent node
    came_from[start_coords] = None
    cost_so_far[start_coords] = 0

    while not frontier.empty():
        current_priority, current_coords = frontier.get()
        # unpacking the tuple current_coords into maze
        current = maze.get(*current_coords)
        print(current)

        if current_coords == goal_coords:
            break

        """
        1. Iterate through the neighbors of the current node.
        2. For each neighbor, calculate its coordinates.
        3. Determine the new path cost to reach this neighbor (current cost to reach the current node + 1)
        4. if the neighbor has not been visited or the new cost is lower, update the open list to include the neighbor
        """
        for next in maze.neighbors(current):
            next_coords = (next.x, next.y)
            new_cost = cost_so_far[current_coords] + 1
            if next_coords not in cost_so_far or new_cost < cost_so_far[next_coords]:
                cost_so_far[next_coords] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put((priority, next_coords))
                came_from[next_coords] = current_coords

    return came_from

def generate_vision_maze(agent: Cell, actual_maze: Maze) -> Maze:
    # when we start, we assume that there are no walls in the maze at all (except for at maze boundaries)
    vision_maze = Maze(WINSIZE)
    vision_maze.grid = [[Wall(x, y, vision_maze) for y in range(vision_maze.h)] for x in range(vision_maze.w)] # initializes all cells to Wall
    for x in range(1, vision_maze.w-1):
        for y in range(1, vision_maze.h-1):
            vision_maze.grid[x][y] = Fog(x,y,vision_maze) # makes all cells except boundary cells Fog

    
    # update maze with information observed by agent's initial placement
    vision_maze.grid[agent.x][agent.y] = actual_maze.grid[agent.x][agent.y]
    vision_maze = update_vision_maze(agent, actual_maze, vision_maze)
    return vision_maze

def update_vision_maze(agent: Cell, actual_maze: Maze, vision_maze: Maze) -> Maze:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = agent.x + dx, agent.y + dy
        if 0 <= nx < actual_maze.w and 0 <= ny < actual_maze.h:
            vision_maze.grid[nx][ny] = actual_maze.get(nx, ny)
    return vision_maze

def find_priority(start: Cell, goal: Cell, g: dict, big_g_pref: bool) -> int: # not sure if works 100%
    # finding f value
    f = g[(start.x, start.y)] + heuristic(start, goal)
    c = start.maze.w * start.maze.h # constant bigger than any possible generated g value
    # packing a random large number to break remaining ties
    random_tie_breaker = random.randint(1,10**20-1)
    if big_g_pref: priority = (f, c*f-g[(start.x, start.y)], random_tie_breaker) # f value, g tiebreaker, random value tiebreaker
    else: priority = (f, c*f+g[(start.x, start.y)], random_tie_breaker)
    return priority

def compute_path(maze: Maze, goal: Cell, open_list: list, closed_list: list, g: dict, tree: dict, search: dict, counter: int, big_g_pref: bool, backward=False, agent=None) -> int:
    #try:
    if not backward:    
        while open_list != [] and g[(goal.x, goal.y)] > open_list[0][0][0]:
            #print(open_list)
            s_tuple = heapq.heappop(open_list)
            s_coords = s_tuple[1]
            closed_list.append(s_coords)
            for successor in maze.neighbors(maze.get(*s_coords)):
                succ_coords = (successor.x, successor.y) 
                if succ_coords in closed_list: continue
                if search[succ_coords] < counter:
                    g[succ_coords] = float('inf')
                    search[succ_coords] = counter
                if g[succ_coords] > g[s_coords] + 1:
                    g[succ_coords] = g[s_coords] + 1
                    tree[succ_coords] = s_coords
                    for tup in open_list:
                        if succ_coords == tup[1]: 
                            open_list.remove(tup)
                            break
                    open_list.append((find_priority(successor, goal, g, big_g_pref), succ_coords))
        return len(closed_list)
    elif backward:
        while open_list != [] and g[(goal.x, goal.y)] > open_list[0][0][0]:
            #print(open_list)
            s_tuple = heapq.heappop(open_list)
            s_coords = s_tuple[1]
            closed_list.append(s_coords)
            for successor in maze.neighbors(maze.get(*s_coords)):
                succ_coords = (successor.x, successor.y) 
                if succ_coords in closed_list: continue
                if search[succ_coords] < counter:
                    g[succ_coords] = float('inf')
                    search[succ_coords] = counter
                if g[succ_coords] > g[s_coords] + 1:
                    g[succ_coords] = g[s_coords] + 1
                    tree[succ_coords] = s_coords
                    for tup in open_list:
                        if succ_coords == tup[1]: 
                            open_list.remove(tup)
                            break
                    open_list.append((find_priority(successor, goal, g, big_g_pref), succ_coords))
        return len(closed_list)
    #except Exception as e:
    #    print("Exception", e, "occurred. Let's investigate...")

                


def no_longer_in_use(screen, actual_maze: Maze, vision_maze: Maze, start: Cell, goal: Cell):
    pygame.draw.rect(screen, (255, 0, 0), (actual_maze.get(goal.x, goal.y).rect.x, actual_maze.get(goal.x, goal.y).rect.y, Cell.w, Cell.h))
    
    counter = 0 # the value of counter is i during the i-th A* search
    search = {} # key: cell coords tup, value: i if cell c was generated last by the i-th A* search
    g = {} # key: cell coords tup, value: int cost to get there from start cell
    tree = {} # key: cell coords tup, value: that cell's parent cell coords tup (useful for retracing paths)

    # initialize the search value of all cells to 0
    for x in range(vision_maze.w):
        for y in range(vision_maze.h):
            search[(x, y)] = 0

    agent = (start.x, start.y)
    while agent != (goal.x,goal.y):
        counter += 1
        g[agent] = 0
        search[agent] = counter
        g[(goal.x, goal.y)] = float('inf')
        search[(goal.x, goal.y)] = counter
        open_list = [] # list functioning as pq. highest priority at index 0. Contains tuple like ((priority tup), (cell cords tup))
        closed_list = [] # a list of already visited cell coordinate tuples
        heapq.heappush(open_list, (find_priority(start, goal, g), agent)) # put start into open list with f value as priority
        compute_path(vision_maze, goal, open_list, closed_list, g, tree, search, counter)
        if open_list == []:
            print("I cannot reach the target")
            return
        """
        follow the tree pointers from goal to start and then move the agent along the resulting path from start to goal
        until it reaches goal or one or more action costs on the path increase.
        """
        # *** CREATE THE PATH ***
        path = [] # a list containing path cell coord tuples
        path.append((goal.x, goal.y))
        parent_coords = tree[(goal.x, goal.y)]
        # retrace the tree list to find the path from goal to start
        while parent_coords != agent:
            path.append(parent_coords)
            parent_coords = tree[parent_coords]
        path.append(parent_coords)
        path.reverse() # now the path is in order from start to goal

        # *** FOLLOW PATH UNTIL OBSTACLE ***
        # *** UPDATE VISION MAZE WITH NEW INFO ***
        for i in range(len(path)): # assumes that path[0] is the start and thus cannot be a wall
            agent = vision_maze.get(*path[i]) # agent will be moving along the calculated path until finding an obstacle
            if isinstance(vision_maze.get(*path[i]), Wall):
                pygame.draw.rect(screen, (96, 96, 96),(actual_maze.get(*path[i]).rect.x, actual_maze.get(*path[i]).rect.y, Cell.w, Cell.h)) # obstacles agent ran into
                pygame.display.update()
                pygame.event.pump()
                pygame.time.delay(100)
                pygame.draw.rect(screen, (0, 0, 0),(actual_maze.get(*path[i]).rect.x, actual_maze.get(*path[i]).rect.y, Cell.w, Cell.h)) # obstacles agent ran into
                pygame.display.update()
                pygame.event.pump()
                agent = vision_maze.get(*path[i-1])
                agent = (agent.x, agent.y)
                break     
            vision_maze = update_vision_maze(agent, actual_maze, vision_maze)
            agent = (agent.x, agent.y)
            pygame.draw.rect(screen, (51, 153, 255),(actual_maze.get(agent[0],agent[1]).rect.x, actual_maze.get(agent[0],agent[1]).rect.y, Cell.w, Cell.h)) # current agent location
            if i > 0: pygame.draw.rect(screen, (0, 0, 180), (actual_maze.get(*path[i-1]).rect.x, actual_maze.get(*path[i-1]).rect.y, Cell.w, Cell.h)) # where has agent been
            if path[i-1] == (start.x, start.y): pygame.draw.rect(screen, (0, 255, 0), (actual_maze.get(*path[i-1]).rect.x, actual_maze.get(*path[i-1]).rect.y, Cell.w, Cell.h)) # start stays same color even if agent has been there
            
            pygame.display.update()
            pygame.event.pump()
            pygame.time.delay(100)
    print("I reached the target")

# will return the number of expanded cells
def repeated_forward_a_star_search_experiment(screen, actual_maze: Maze, vision_maze: Maze, start: Cell, goal: Cell, big_g_pref: bool) -> int:
    pygame.draw.rect(screen, (255, 0, 0), (actual_maze.get(goal.x, goal.y).rect.x, actual_maze.get(goal.x, goal.y).rect.y, Cell.w, Cell.h))
    
    counter = 0 # the value of counter is i during the i-th A* search
    search = {} # key: cell coords tup, value: i if cell c was generated last by the i-th A* search
    g = {} # key: cell coords tup, value: int cost to get there from start cell
    tree = {} # key: cell coords tup, value: that cell's parent cell coords tup (useful for retracing paths)
    num_expanded_cells = 0 # the total number of expanded cells

    # initialize the search value of all cells to 0
    for x in range(vision_maze.w):
        for y in range(vision_maze.h):
            search[(x, y)] = 0

    agent = (start.x, start.y)
    while agent != (goal.x,goal.y):
        counter += 1
        g[agent] = 0
        search[agent] = counter
        g[(goal.x, goal.y)] = float('inf')
        search[(goal.x, goal.y)] = counter
        open_list = [] # list functioning as pq. highest priority at index 0. Contains tuple like ((priority tup), (cell cords tup))
        closed_list = [] # a list of already visited cell coordinate tuples
        heapq.heappush(open_list, (find_priority(vision_maze.get(*agent), goal, g, big_g_pref), agent)) # put start into open list with f value as priority
        num_expanded_cells += compute_path(vision_maze, goal, open_list, closed_list, g, tree, search, counter, big_g_pref)
        if open_list == []:
            print("I cannot reach the target")
            return num_expanded_cells
        """
        follow the tree pointers from goal to start and then move the agent along the resulting path from start to goal
        until it reaches goal or one or more action costs on the path increase.
        """
        # *** CREATE THE PATH ***
        path = [] # a list containing path cell coord tuples
        path.append((goal.x, goal.y))
        parent_coords = tree[(goal.x, goal.y)]
        # retrace the tree list to find the path from goal to start
        #print(tree)
        while parent_coords != agent:
            path.append(parent_coords)
            parent_coords = tree[parent_coords]
        path.append(parent_coords)
        path.reverse() # now the path is in order from start to goal
        #return
        # *** FOLLOW PATH UNTIL OBSTACLE ***
        # *** UPDATE VISION MAZE WITH NEW INFO ***
        for i in range(len(path)): # assumes that path[0] is the start and thus cannot be a wall
            agent = vision_maze.get(*path[i]) # agent will be moving along the calculated path until finding an obstacle
            if isinstance(vision_maze.get(*path[i]), Wall):
                pygame.draw.rect(screen, (96, 96, 96),(actual_maze.get(*path[i]).rect.x, actual_maze.get(*path[i]).rect.y, Cell.w, Cell.h)) # obstacles agent ran into
                pygame.display.update()
                pygame.event.pump()
                #pygame.time.delay(100)
                pygame.draw.rect(screen, (0, 0, 0),(actual_maze.get(*path[i]).rect.x, actual_maze.get(*path[i]).rect.y, Cell.w, Cell.h)) # obstacles agent ran into
                pygame.display.update()
                pygame.event.pump()
                agent = vision_maze.get(*path[i-1])
                agent = (agent.x, agent.y)
                break     
            vision_maze = update_vision_maze(agent, actual_maze, vision_maze)
            agent = (agent.x, agent.y)
            pygame.draw.rect(screen, (51, 153, 255),(actual_maze.get(agent[0],agent[1]).rect.x, actual_maze.get(agent[0],agent[1]).rect.y, Cell.w, Cell.h)) # current agent location
            if i > 0: pygame.draw.rect(screen, (0, 0, 180), (actual_maze.get(*path[i-1]).rect.x, actual_maze.get(*path[i-1]).rect.y, Cell.w, Cell.h)) # where has agent been
            if path[i-1] == (start.x, start.y): pygame.draw.rect(screen, (0, 255, 0), (actual_maze.get(*path[i-1]).rect.x, actual_maze.get(*path[i-1]).rect.y, Cell.w, Cell.h)) # start stays same color even if agent has been there
            
            pygame.display.update()
            pygame.event.pump()
            #pygame.time.delay(100)
    print("I reached the target")
    return num_expanded_cells


# will return the number of expanded cells
def repeated_backward_a_star_search_experiment(screen, actual_maze: Maze, vision_maze: Maze, start: Cell, goal: Cell, big_g_pref: bool) -> int:
    pygame.draw.rect(screen, (255, 0, 0), (actual_maze.get(goal.x, goal.y).rect.x, actual_maze.get(goal.x, goal.y).rect.y, Cell.w, Cell.h))
    
    counter = 0 # the value of counter is i during the i-th A* search
    search = {} # key: cell coords tup, value: i if cell c was generated last by the i-th A* search
    g = {} # key: cell coords tup, value: int cost to get there from start cell
    tree = {} # key: cell coords tup, value: that cell's parent cell coords tup (useful for retracing paths)
    num_expanded_cells = 0 # the total number of expanded cells

    # initialize the search value of all cells to 0
    for x in range(vision_maze.w):
        for y in range(vision_maze.h):
            search[(x, y)] = 0

    agent = (start.x, start.y)
    while agent != (goal.x,goal.y):
        counter += 1
        g[agent] = float('inf')
        search[agent] = counter
        g[(goal.x, goal.y)] = 0
        search[(goal.x, goal.y)] = counter
        open_list = [] # list functioning as pq. highest priority at index 0. Contains tuple like ((priority tup), (cell cords tup))
        closed_list = [] # a list of already visited cell coordinate tuples
        heapq.heappush(open_list, (find_priority(goal, vision_maze.get(*agent), g, big_g_pref), (goal.x, goal.y))) # put goal into open list with f value as priority
        num_expanded_cells += compute_path(vision_maze, vision_maze.get(*agent), open_list, closed_list, g, tree, search, counter, big_g_pref, True, vision_maze.get(*agent)) # start our pathfind from goal to agent
        if open_list == []:
            print("I cannot reach the target")
            return num_expanded_cells
        """
        follow the tree pointers from goal to start and then move the agent along the resulting path from start to goal
        until it reaches goal or one or more action costs on the path increase.
        """
        # *** CREATE THE PATH ***
        path = [] # a list containing path cell coord tuples
        path.append(agent)
        parent_coords = tree[agent]
        # retrace the tree list to find the path from goal to start
        while parent_coords != (goal.x,goal.y):
            path.append(parent_coords)
            parent_coords = tree[parent_coords]
        path.append(parent_coords)
        #return
        #path.reverse()

        # *** FOLLOW PATH UNTIL OBSTACLE ***
        # *** UPDATE VISION MAZE WITH NEW INFO ***
        for i in range(len(path)): # assumes that path[0] is the start and thus cannot be a wall
            agent = vision_maze.get(*path[i]) # agent will be moving along the calculated path until finding an obstacle
            if isinstance(vision_maze.get(*path[i]), Wall):
                pygame.draw.rect(screen, (96, 96, 96),(actual_maze.get(*path[i]).rect.x, actual_maze.get(*path[i]).rect.y, Cell.w, Cell.h)) # obstacles agent ran into
                pygame.display.update()
                pygame.event.pump()
                #pygame.time.delay(500)
                pygame.draw.rect(screen, (0, 0, 0),(actual_maze.get(*path[i]).rect.x, actual_maze.get(*path[i]).rect.y, Cell.w, Cell.h)) # obstacles agent ran into
                pygame.display.update()
                pygame.event.pump()
                agent = vision_maze.get(*path[i-1])
                agent = (agent.x, agent.y)
                break     
            vision_maze = update_vision_maze(agent, actual_maze, vision_maze)
            agent = (agent.x, agent.y)
            pygame.draw.rect(screen, (51, 153, 255),(actual_maze.get(agent[0],agent[1]).rect.x, actual_maze.get(agent[0],agent[1]).rect.y, Cell.w, Cell.h)) # current agent location
            if i > 0: pygame.draw.rect(screen, (0, 0, 180), (actual_maze.get(*path[i-1]).rect.x, actual_maze.get(*path[i-1]).rect.y, Cell.w, Cell.h)) # where has agent been
            if path[i-1] == (start.x, start.y): pygame.draw.rect(screen, (0, 255, 0), (actual_maze.get(*path[i-1]).rect.x, actual_maze.get(*path[i-1]).rect.y, Cell.w, Cell.h)) # start stays same color even if agent has been there
            
            pygame.display.update()
            pygame.event.pump()
            #pygame.time.delay(500)
    print("I reached the target")
    return num_expanded_cells





"""
from the "closed list" (came_from) I can reconstruct the path from the start to the goal 
by following the parent pointers from the goal to the start.
"""

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


def draw_path(screen, path):
    for node in path:
        pygame.draw.rect(screen, (0, 255, 0),
                         (node.rect.x, node.rect.y, Cell.w, Cell.h))


def draw_maze(screen):
    maze = Maze(WINSIZE)
    maze.generate(screen, True)

    start = maze.get(1, 1)
    goal = maze.get(maze.w - 2, maze.h - 2)
    vision_maze = generate_vision_maze(start, maze)
    #repeated_forward_a_star_search(screen, maze, vision_maze, start, goal)
    
    # came_from = a_star_search(maze, start, goal)
    # path = reconstruct_path(came_from, start, goal, maze)
    # draw_path(screen, path)

def maze_list_experiment_backup(screen, maze_list):
    expanded_cells_list_big_g = []
    for maze in maze_list:
        start = maze.get(1,1)
        goal = maze.get(maze.w - 2, maze.h - 2)
        vision_maze = generate_vision_maze(start, maze)       
        expanded_cells_list_big_g.append(repeated_forward_a_star_search_big_g(screen, maze, vision_maze, start, goal))

def maze_reset(screen, maze):
    for x in range(maze.w):
        for y in range(maze.h):
            if isinstance(maze.grid[x][y], Cell) and screen.get_at((maze.grid[x][y].rect.x, maze.grid[x][y].rect.y)) == (0, 0, 180) or screen.get_at((maze.grid[x][y].rect.x, maze.grid[x][y].rect.y)) == (51, 153, 255):
                pygame.draw.rect(screen, (255,255,255), (maze.grid[x][y].rect.x, maze.grid[x][y].rect.y, Cell.w, Cell.h))
                pygame.display.update()
                pygame.event.pump()

def make_maze_unsolvable(screen, actual_maze, goal: Cell) -> Maze:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = goal.x + dx, goal.y + dy
        if 0 <= nx < actual_maze.w and 0 <= ny < actual_maze.h and not isinstance(actual_maze.get(nx, ny), Wall):
            actual_maze.grid[nx][ny] = Wall(nx,ny,actual_maze)
            pygame.draw.rect(screen, (0, 0, 0), (actual_maze.grid[nx][ny].rect.x, actual_maze.grid[nx][ny].rect.y, Cell.w, Cell.h))
            pygame.display.update()
            pygame.event.pump()
    return actual_maze

def big_g_vs_little_g(screen, num_mazes):
    expanded_cells_list_big_g = []
    expanded_cells_list_little_g = []
    for i in range(int(num_mazes)):
        print("Testing maze", i+1, "out of",num_mazes,"!")
        pygame.display.set_caption("Forward Repeated A* (Big g preference)")
        maze = Maze(WINSIZE)
        maze.generate(screen, True)
        start = maze.get(1,1)
        goal = maze.get(maze.w - 2, maze.h - 2)
        
        vision_maze = generate_vision_maze(start, maze)       
        expanded_cells_list_big_g.append(repeated_forward_a_star_search_experiment(screen, maze, vision_maze, start, goal, big_g_pref=True))
        pygame.display.set_caption("Resetting Maze...")        
        maze_reset(screen, maze)
        pygame.display.set_caption("Forward Repeated A* (Little g preference)")
        vision_maze = generate_vision_maze(start, maze) 
        expanded_cells_list_little_g.append(repeated_forward_a_star_search_experiment(screen, maze, vision_maze, start, goal, big_g_pref=False))

    print("sum of expanded cells list big g:", sum(expanded_cells_list_big_g))
    print("len of expanded cells list big g:", len(expanded_cells_list_big_g))
    big_g_average = sum(expanded_cells_list_big_g)/len(expanded_cells_list_big_g)
    print("average number of expanded cells for repeated forward A* on",num_mazes, maze.w-2,"x",maze.h-2, "graphs with larger g-value tiebreaking:", big_g_average)
    
    print("sum of expanded cells list little g:", sum(expanded_cells_list_little_g))
    print("len of expanded cells list little g:", len(expanded_cells_list_little_g))
    little_g_average = sum(expanded_cells_list_little_g)/len(expanded_cells_list_little_g)
    print("average number of expanded cells for repeated forward A* on", num_mazes, maze.w-2,"x",maze.h-2, "graphs with smaller g-value tiebreaking:", little_g_average)

def forward_vs_backward(screen, num_mazes):
    expanded_cells_list_forward = [] 
    expanded_cells_list_backward = []
    for i in range(int(num_mazes)):
        print("Testing maze", i+1, "out of",num_mazes,"!")
        pygame.display.set_caption("Repeated Forward A*")
        maze = Maze(WINSIZE)
        maze.generate(screen, True)
        
        start = maze.get(1,1) 
        goal = maze.get(maze.w - 2, maze.h - 2)
        if i % 2 == 0: maze = make_maze_unsolvable(screen, maze, goal)

        vision_maze = generate_vision_maze(start, maze)
        expanded_cells_list_forward.append(repeated_forward_a_star_search_experiment(screen,maze,vision_maze,start,goal,big_g_pref=True))

        pygame.display.set_caption("Resetting Maze...")
        maze_reset(screen, maze)

        pygame.display.set_caption("Repeated Backward A*")
        start = maze.get(1,1)
        goal = maze.get(maze.w - 2, maze.h - 2)
        vision_maze = generate_vision_maze(start,maze)
        expanded_cells_list_backward.append(repeated_backward_a_star_search_experiment(screen,maze,vision_maze,start,goal,big_g_pref=True))
    
    print("sum of expanded cells list forward repeated A*:", sum(expanded_cells_list_forward))
    print("len of expanded cells list forward repeated A*:", len(expanded_cells_list_forward))
    forward_average = sum(expanded_cells_list_forward)/len(expanded_cells_list_forward)
    print("average number of expanded cells for repeated forward A* on",num_mazes, maze.w-2,"x",maze.h-2, "graphs with larger g-value tiebreaking:", forward_average)
    
    print("sum of expanded cells list backward repeated A*:", sum(expanded_cells_list_backward))
    print("len of expanded cells list backward repeated A*:", len(expanded_cells_list_backward))
    backward_average = sum(expanded_cells_list_backward)/len(expanded_cells_list_backward)
    print("average number of expanded cells for repeated backward A* on", num_mazes, maze.w-2,"x",maze.h-2, "graphs with larger g-value tiebreaking:", backward_average)


"""Winsize sets the dimension of the maze. Make sure it's an odd number. """
WINSIZE = (Cell.w * 51, Cell.h * 51) 

def main():
    pygame.init()
    scr_inf = pygame.display.Info()
    os.environ['SDL_VIDEO_WINDOW_POS'] = '{}, {}'.format(scr_inf.current_w // 2 - WINSIZE[0] // 2,
                                                         scr_inf.current_h // 2 - WINSIZE[1] // 2)
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('Maze with A* Pathfinding')
    screen.fill((0, 0, 0))

    clock = pygame.time.Clock()


    num_mazes = input("How many mazes would you like to generate? ")
    #print("OK! Generating", command, "mazes...")
    #maze_list = generate_maze_list(screen, command)
    #print("Finished!")
    command = input("What would you like to do? ")
    if command == "2":
        big_g_vs_little_g(screen, num_mazes)
    elif command == "3":
        forward_vs_backward(screen, num_mazes)

    #draw_maze(screen)

    done = False
    while not done:
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                done = True

        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    main()
