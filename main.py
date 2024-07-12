from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import random
import matplotlib.patches as mpatches

WIDTH = 12
HEIGHT = 12
EPISODES = 1000
EPSILON = 0.1
LEARNING_RATE = 0.1
GAMMA = 0.9
ACTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def reward(s):
    x, y = s
    if x == WIDTH - 1 and y == HEIGHT / 2:
        return 100
    if y < HEIGHT - 1 and y > 0 and x == WIDTH / 2:
        return -10
    # return 0
    return -1
    # dx, dy = x - (WIDTH - 1), y - HEIGHT / 2
    # if dx == 0 and dy == 0:
    #     return 10
    # return 1 / np.sqrt(dx * dx + dy * dy)

def take_action(s, action):
    cx, cy = s
    ax, ay = ACTIONS[action]

    if cx + ax < 0:
        ax = 0
    if cx + ax >= WIDTH:
        ax = 0
    if cy + ay < 0:
        ay = 0
    if cy + ay >= HEIGHT:
        ay = 0

    new_state = cx + ax, cy + ay
    r = reward(new_state)
    done = cx == WIDTH - 1 and cy == HEIGHT/2
    return new_state, r, done

def select_action(Q, s):
    x, y = s
    if random.random() < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)
    return np.argmax(Q[x * HEIGHT + y])

Q = np.random.rand(WIDTH * HEIGHT, len(ACTIONS))
HIS = np.zeros((HEIGHT, WIDTH))
REWARDS = np.array([])
for _ in range(EPISODES):
    total_reward = 0
    done = False
    s = (0, int(HEIGHT/2))
    i = 0
    while not done:
        i += 1
        x, y = s
        HIS[y, x] += 1
        a = select_action(Q, s)
        _s, r, done = take_action(s, a)
        _x, _y = _s
        Q[x * HEIGHT + y, a] = Q[x * HEIGHT + y, a] + LEARNING_RATE * (r + GAMMA * np.max(Q[_x * HEIGHT + _y]) - Q[x * HEIGHT + y, a])
        s = _s
        total_reward += r
    REWARDS = np.append(REWARDS, total_reward)


mean_Q = np.mean(REWARDS.reshape(-1, 5), axis=1)

fig, axs = plt.subplots(1)
axs.set_title('Q-Learning')
axs.imshow(HIS, cmap='viridis', interpolation='nearest')
plt.savefig('final.png')
