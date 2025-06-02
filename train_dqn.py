import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
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

class DQNTrainer:
    def __init__(self, grid_size, curriculum):
        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(7, 4).to(self.device)
        self.target_model = DQN(7, 4).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.curriculum = curriculum

    def get_state(self, env):
        head = env.snake1[0]
        food = env.food
        obs = env.obstacles

        d_head_obs = min((abs(head[0] - x) + abs(head[1] - y)) for (x, y) in obs) if obs else 0
        d_food_obs = min((abs(food[0] - x) + abs(food[1] - y)) for (x, y) in obs) if obs else 0

        return np.array([
            head[0] / self.grid_size,
            head[1] / self.grid_size,
            food[0] / self.grid_size,
            food[1] / self.grid_size,
            d_head_obs / self.grid_size,
            d_food_obs / self.grid_size,
            len(obs) / 100
        ], dtype=np.float32)

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

    def save_checkpoint(self, episode, path="checkpoints"):
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        filepath = os.path.join(path, f"checkpoint_ep{episode}.pt")
        torch.save(checkpoint, filepath)
        torch.save(checkpoint, os.path.join(path, "checkpoint_latest.pt"))  # 最新版本覆蓋

    def load_checkpoint(self, filename="checkpoints/checkpoint_latest.pt"):
        if not os.path.exists(filename):
            print("No checkpoint found. Starting from scratch.")
            return 0
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Loaded checkpoint from {filename} at episode {checkpoint['episode']}")
        return checkpoint['episode']

    def train(self, start_ep=0, max_ep=None):
        directions = [(1,0), (0,1), (-1,0), (0,-1)]
        total_episodes = sum(episodes for _, _, episodes in self.curriculum)
        ep_counter = start_ep

        for map_obstacles, difficulty, num_eps in self.curriculum:
            for _ in range(num_eps):
                if max_ep is not None and ep_counter >= max_ep:
                    return

                env = SnakeGame(self.grid_size, self.grid_size, max_apples=10, difficulty=difficulty, obstacle_map=map_obstacles)
                env.reset()
                state = self.get_state(env)
                total_reward = 0
                prev_dist = abs(env.snake1[0][0] - env.food[0]) + abs(env.snake1[0][1] - env.food[1])
                max_steps = 1000  # Limit steps to prevent infinite loops
                done = False
                step=0
                while not done and step < max_steps:
                    step += 1
                    action = self.get_action(state)
                    env.update_snake1(directions[action])
                    next_state = self.get_state(env)

                    head = env.snake1[0]
                    if head == env.food:
                        reward = 1
                    elif head in env.obstacles:
                        reward = -1
                    else:
                        new_dist = abs(head[0] - env.food[0]) + abs(head[1] - env.food[1])
                        reward = 0.05 if new_dist < prev_dist else -0.02
                        prev_dist = new_dist

                    done = env.is_game_over()
                    self.remember(state, action, reward, next_state, done)
                    self.train_step()
                    state = next_state
                    total_reward += reward

                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                if ep_counter % 20 == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
                if ep_counter % 100 == 0:
                    self.save_checkpoint(ep_counter)

                print(f"Episode {ep_counter+1}/{total_episodes} Reward: {total_reward:.2f} Epsilon: {self.epsilon:.2f}")
                ep_counter += 1

                
        torch.save(self.model.state_dict(), "dqn_snake_hard.pth")
        print("Training completed and model saved.")

def curriculum(grid_size):
    return [
        (empty_map, "easy", 20),
        (simple_wall_map, "hard", 200),
        (cross_wall_map, "hard", 200),
    ]


def training_procedure(grid_size=30, curriculum_fn=None):
    if curriculum_fn is None:
        raise ValueError("You must provide a curriculum function")

    trainer = DQNTrainer(grid_size=grid_size, curriculum=curriculum_fn(grid_size))

    # Phase 1: initial training
    print("\n✨ Initial training (20 episodes)")
    trainer.train(start_ep=0, max_ep=20)
    trainer.save_checkpoint(20)
    print(f"\n✅ Checkpoint saved at episode 20")

    # Phase 2: resume training
    print("\n🔄 Resume training from last checkpoint")
    trainer2 = DQNTrainer(grid_size=grid_size, curriculum=curriculum_fn(grid_size))
    last_ep = trainer2.load_checkpoint()
    trainer2.train(start_ep=last_ep + 1)

if __name__ == "__main__":
    training_procedure(grid_size=30, curriculum_fn=curriculum)
