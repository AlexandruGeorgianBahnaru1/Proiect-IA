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


class FrozenLakeGUI:
    def __init__(self, root, env, Q, image_paths):
        self.root = root
        self.env = env
        self.Q = Q
        self.canvas_size = 400
        self.cell_size = self.canvas_size // env.grid_size
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.restart_button = tk.Button(root, text="Restart", command=self.restart)
        self.restart_button.pack()

        self.images = {}
        for key, path in image_paths.items():
            image = Image.open(path)
            resized_image = image.resize((self.cell_size, self.cell_size))
            tk_image = ImageTk.PhotoImage(resized_image)
            self.images[key] = tk_image

        self.restart()

    def draw_lake(self):
        self.canvas.delete("all")
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                x, y = j * self.cell_size, i * self.cell_size
                cell = self.env.lake[i][j]

                if cell == 'S':
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.images['start'])
                elif cell == 'H':
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.images['hole'])
                elif cell == 'R':
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.images['carrot'])
                elif cell == 'F':
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.images['ice'])
                elif cell == 'G':
                    self.canvas.create_image(x, y, anchor=tk.NW, image=self.images['goal'])

        x, y = self.env.state[1] * self.cell_size, self.env.state[0] * self.cell_size
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.images['snowman'])

    def restart(self):
        self.env.reset()
        self.done = False
        self.run_agent()

    def run_agent(self):
        if not self.done:
            state_index = self.env.get_state_index()
            action = np.argmax(self.Q[state_index])
            _, _, self.done = self.env.step(action)
            self.draw_lake()
            self.root.after(300, self.run_agent)
        else:
            print("Episod terminat")



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


image_paths = {
    'hole': 'hole.png',
    'carrot': 'carrot.png',
    'ice': 'ice.png',
    'snowman': 'snowman.png',
    'start': 'start.png',
    'goal': 'goal.png'
}

def test_q_learning():
    env = FrozenLake(grid_size=7)
    state = env.reset()
    done = False
    steps = 0
    max_steps = 100
    while not done and steps < max_steps:
        action = np.argmax(Q[state])
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1
    assert done, "Agentul nu a ajuns la destinatie."
    assert reward == 1, "Agentul nu a atins starea de succes."



print("Testarea agentului...")
test_q_learning()
print("Test trecut cu succes!")


root = tk.Tk()
root.title("Frozen Lake 7x7")
app = FrozenLakeGUI(root, env, Q, image_paths)
root.mainloop()
