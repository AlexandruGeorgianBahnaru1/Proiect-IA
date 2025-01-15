from PIL import Image, ImageTk
import numpy as np
import random
import tkinter as tk


class FrozenLake:
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        self.reset()
        self.actions = {
            0: (-1, 0),  # sus
            1: (1, 0),  # jos
            2: (0, -1),  # stanga
            3: (0, 1)  # dreapta
        }

    def reset(self):
        self.lake = [
            ['S', 'F', 'F', 'F', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'F', 'H', 'F', 'F'],
            ['F', 'F', 'F', 'H', 'F', 'F', 'H'],
            ['F', 'H', 'F', 'F', 'F', 'H', 'F'],
            ['R', 'F', 'F', 'H', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'F', 'H', 'F', 'H'],
            ['F', 'F', 'H', 'F', 'F', 'F', 'G']
        ]
        self.state = (0, 0)
        self.done = False
        return self.get_state_index()

    def get_state_index(self):
        return self.state[0] * self.grid_size + self.state[1]

    def step(self, action):
        move = self.actions[action]
        new_row = max(0, min(self.grid_size - 1, self.state[0] + move[0]))
        new_col = max(0, min(self.grid_size - 1, self.state[1] + move[1]))
        self.state = (new_row, new_col)

        cell = self.lake[new_row][new_col]
        if cell == 'H':
            print("Ai cazut intr-o groapa")
            self.done = True
            reward = -1
        elif cell == 'G':
            print("Ai ajuns la destinatie")
            self.done = True
            reward = 1
        elif cell == 'R':
            reward = -0.05
        else:
            reward = -0.1

        return self.get_state_index(), reward, self.done

alpha, gamma, epsilon = 0.8, 0.95, 0.1
num_episodes = 3000
grid_size = 7
num_states = grid_size * grid_size
num_actions = 4
Q = np.zeros((num_states, num_actions))

env = FrozenLake(grid_size=grid_size)
for _ in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(Q[state])
        next_state, reward, done = env.step(action)
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state