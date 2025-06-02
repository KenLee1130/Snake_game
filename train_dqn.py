import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cupy as cp
import random
from collections import deque
from numba import jit
from game_env import SnakeGame
from maps import empty_map, simple_wall_map, cross_wall_map

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

@jit(nopython=True)
def compute_reward(snake_head_x, snake_head_y, food_x, food_y):
    return 1.0 if (snake_head_x == food_x and snake_head_y == food_y) else -0.01

class DQNTrainer:
    def __init__(self, grid_size, episodes=600):
        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(6, 4).to(self.device)
        self.target_model = DQN(6, 4).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.episodes = episodes

    def get_state(self, env):
        head = env.snake1[0]
        food = env.food
        dx = food[0] - head[0]
        dy = food[1] - head[1]
        dx /= self.grid_size
        dy /= self.grid_size
        state_cp = cp.array([
            head[0] / self.grid_size,
            head[1] / self.grid_size,
            dx,
            dy,
            int(env.difficulty == "hard"),
            len(env.obstacles) / 100
        ], dtype=cp.float32)
        return cp.asnumpy(state_cp)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
        targets = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_map(self, ep):
        if ep < self.episodes * 0.33:
            return empty_map(), "easy"
        elif ep < self.episodes * 0.66:
            return simple_wall_map(), "hard"
        else:
            return cross_wall_map(), "hard"

    def train(self):
        for ep in range(self.episodes):
            map_obstacles, difficulty = self.select_map(ep)
            env = SnakeGame(self.grid_size, self.grid_size, max_apples=10, difficulty=difficulty, fixed_obstacles=map_obstacles)
            env.reset()
            state = self.get_state(env)
            total_reward = 0
            done = False
            while not done:
                action = self.get_action(state)
                directions = [(1,0), (0,1), (-1,0), (0,-1)]
                env.update_snake1(directions[action])
                next_state = self.get_state(env)
                reward = compute_reward(env.snake1[0][0], env.snake1[0][1], env.food[0], env.food[1])
                done = env.is_game_over()
                self.remember(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if ep % 20 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            print(f"Episode {ep+1} Reward: {total_reward:.2f} Epsilon: {self.epsilon:.2f}")

        torch.save(self.model, "dqn_snake_hard.pth")
        print("Training completed and model saved.")

if __name__ == "__main__":
    trainer = DQNTrainer(grid_size=30, episodes=600)
    trainer.train()
