import pygame
from pygame.locals import *
from random import choice
import os


class Cell(pygame.sprite.Sprite):
    w, h = 16, 16

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

    def draw(self, screen):
        screen.blit(self.image, self.rect)


class Wall(Cell):
    def __init__(self, x, y, maze):
        super(Wall, self).__init__(x, y, maze)
        self.image.fill((0, 0, 0))
        self.type = 0


class Maze:
    def __init__(self, size):
        self.w, self.h = size[0] // Cell.w, size[1] // Cell.h
        self.grid = [[Wall(x, y, self) for y in range(self.h)]
                     for x in range(self.w)]

    def get(self, x, y):
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


WINSIZE = (Cell.w * 41, Cell.h * 41)


def draw_maze(screen):
    maze = Maze(WINSIZE)
    maze.generate(screen, True)


def main():
    pygame.init()
    scr_inf = pygame.display.Info()
    os.environ['SDL_VIDEO_WINDOW_POS'] = '{}, {}'.format(scr_inf.current_w // 2 - WINSIZE[0] // 2,
                                                         scr_inf.current_h // 2 - WINSIZE[1] // 2)
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('Maze')
    screen.fill((0, 0, 0))

    clock = pygame.time.Clock()
    draw_maze(screen)

    done = 0
    while not done:
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                done = 1

        pygame.display.update()
        clock.tick()


if __name__ == '__main__':
    main()
