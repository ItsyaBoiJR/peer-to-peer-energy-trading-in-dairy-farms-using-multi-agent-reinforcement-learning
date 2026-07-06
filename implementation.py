import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Simulate the environment
class EnergyTradingEnv:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.state_dim = 3  # Example: [energy demand, battery level, market price]
        self.action_dim = 3  # Example: [buy, sell, hold]
        self.reset()

    def reset(self):
        self.states = np.random.rand(self.num_agents, self.state_dim)
        return self.states

    def step(self, actions):
        rewards = []
        next_states = []
        dones = []
        for i, action in enumerate(actions):
            reward = np.random.rand() - 0.5  # Random reward for simplicity
            next_state = np.random.rand(self.state_dim)
            done = np.random.rand() < 0.1  # Random termination condition
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.array(next_states), np.array(rewards), np.array(dones)

if __name__ == '__main__':
    num_agents = 5
    env = EnergyTradingEnv(num_agents)
    agents = [DQNAgent(env.state_dim, env.action_dim) for _ in range(num_agents)]

    episodes = 100
    for episode in range(episodes):
        states = env.reset()
        total_rewards = np.zeros(num_agents)

        for t in range(50):  # Max steps per episode
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones = env.step(actions)

            for i, agent in enumerate(agents):
                agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                total_rewards[i] += rewards[i]

            states = next_states

            for agent in agents:
                agent.replay()

        for agent in agents:
            agent.update_target_model()

        print(f"Episode {episode + 1}: Total Rewards: {total_rewards}")