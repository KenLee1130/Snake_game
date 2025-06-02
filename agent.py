import pygame

class HumanAgent:
    def __init__(self):
        self.last_dir = (1, 0)

    def update_direction(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and self.last_dir != (0, 1):
                self.last_dir = (0, -1)
            elif event.key == pygame.K_DOWN and self.last_dir != (0, -1):
                self.last_dir = (0, 1)
            elif event.key == pygame.K_LEFT and self.last_dir != (1, 0):
                self.last_dir = (-1, 0)
            elif event.key == pygame.K_RIGHT and self.last_dir != (-1, 0):
                self.last_dir = (1, 0)

    def get_action(self):
        return self.last_dir

class SimpleAIAgent:
    def get_action(self, snake, food, enemy_snake, obstacles):
        head_x, head_y = snake[0]
        fx, fy = food
        dx = 1 if fx > head_x else -1 if fx < head_x else 0
        dy = 1 if fy > head_y else -1 if fy < head_y else 0
        return (dx, dy)

class DQNAgent:
    def __init__(self, model_path):
        import torch
        from train_dqn import DQN
        self.model = DQN(7, 4)
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_action(self, snake, food, enemy_snake, obstacles):
        # Placeholder: you will implement actual state encoding and forward
        return (0, 1)  # dummy move down
