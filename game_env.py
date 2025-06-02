import pygame
import random

class SnakeGame:
    def __init__(self, width, height, max_apples=10, difficulty="easy", obstacle_map=None):
        self.grid_width = width
        self.grid_height = height
        self.max_apples = max_apples
        self.difficulty = difficulty
        self.obstacle_map = obstacle_map
        self.block_size = 20

        self.reset()

    def reset(self):
        self.snake1 = [(5, 5)]
        self.snake2 = [(15, 15)]
        self.direction1 = (1, 0)
        self.direction2 = (-1, 0)
        self.snake1_dead = False
        self.snake2_dead = False
        self.score1 = 0
        self.score2 = 0
        self.game_over = False
        self.obstacles = self.generate_obstacles() if self.difficulty == "hard" else []
        self.food = self.spawn_food()

    def generate_obstacles(self):
        if self.obstacle_map is not None:
            return self.obstacle_map(self.grid_width, self.grid_height)
        return []

    def spawn_food(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if pos not in self.snake1 and pos not in self.snake2 and pos not in self.obstacles:
                return pos

    def update_snake1(self, direction):
        if self.game_over: return
        if direction: self.direction1 = direction
        head_x, head_y = self.snake1[0]
        dx, dy = self.direction1
        new_head = ((head_x + dx) % self.grid_width, (head_y + dy) % self.grid_height)
        self.move_snake(new_head, self.snake1, snake_id=1)

    def update_snake2(self, direction):
        if self.game_over: return
        if direction: self.direction2 = direction
        head_x, head_y = self.snake2[0]
        dx, dy = self.direction2
        new_head = ((head_x + dx) % self.grid_width, (head_y + dy) % self.grid_height)
        self.move_snake(new_head, self.snake2, snake_id=2)

    def move_snake(self, new_head, snake, snake_id):
        if (
            new_head in snake or
            new_head in self.obstacles
        ):
            self.game_over = True
            if snake_id == 1:
                self.snake1_dead = True
            elif snake_id == 2:
                self.snake2_dead = True
            return

        snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.spawn_food()
            if snake_id == 1:
                self.score1 += 1
            elif snake_id == 2:
                self.score2 += 1
        else:
            snake.pop()

        if self.score1 >= self.max_apples or self.score2 >= self.max_apples:
            self.game_over = True

    def draw(self, screen, mode="human", show_grid=False):
        screen.fill((20, 20, 20))
        font = pygame.font.SysFont("consolas", 20)

        if show_grid:
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                    pygame.draw.rect(screen, (40, 40, 40), rect, 1)

        for ox, oy in self.obstacles:
            pygame.draw.rect(screen, (80, 80, 80), pygame.Rect(ox * self.block_size, oy * self.block_size, self.block_size, self.block_size))

        for i, (x, y) in enumerate(self.snake1):
            color = (0, 255, 100) if i == 0 else (0, 200, 0)
            pygame.draw.rect(screen, color, pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size))

        if mode == "ai":
            for i, (x, y) in enumerate(self.snake2):
                color = (100, 100, 255) if i == 0 else (0, 0, 200)
                pygame.draw.rect(screen, color, pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size))

        fx, fy = self.food
        pygame.draw.rect(screen, (255, 80, 80), pygame.Rect(fx * self.block_size, fy * self.block_size, self.block_size, self.block_size))

        pygame.draw.rect(screen, (30, 30, 30), (0, self.grid_height * self.block_size, screen.get_width(), 40))
        status = f"P1: {self.score1}   " + (f"P2: {self.score2}   " if mode == "ai" else "") + "Press [P] to pause"
        label = font.render(status, True, (255, 255, 255))
        screen.blit(label, (10, self.grid_height * self.block_size + 10))

    def is_game_over(self):
        return self.game_over
