# core/dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=15000) 
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995 
        self.lr = 0.0003 
        self.sync_target_steps = 150 
        self.step_counter = 0

        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.actions = ["BUY", "SELL", "HOLD"]

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions), random.uniform(0.1, 1.0)
        
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)[0]
        self.model.train()
        
        best_action_idx = q_values.argmax().item()
        action = self.actions[best_action_idx]
        
        q_np = q_values.numpy()
        exp_q = np.exp(q_np - np.max(q_np))
        probabilities = exp_q / exp_q.sum()
        confidence = probabilities[best_action_idx]

        return action, confidence

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = [self.actions.index(b[1]) for b in batch]
        rewards = torch.FloatTensor(np.array([b[2] for b in batch]))
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor(np.array([b[4] for b in batch], dtype=np.float32))

        q_values = self.model(states)
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]

        q_targets = q_values.clone()
        for i in range(batch_size):
            target_val = rewards[i]
            if not dones[i]:
                target_val += self.gamma * max_next_q_values[i]
            q_targets[i][actions[i]] = target_val

        loss = self.loss_fn(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_counter += 1
        if self.step_counter % self.sync_target_steps == 0:
            self.update_target_model()

    def save(self, filename="models/dqn_model.pt"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="models/dqn_model.pt"):
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()

