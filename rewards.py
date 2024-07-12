from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import random
import matplotlib.patches as mpatches

WIDTH = 12
HEIGHT = 4
EPISODES = 250
EPSILON = 0.15
LEARNING_RATE = 0.1
GAMMA = 0.9
ACTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def reward(s):
    x, y = s
    if x == WIDTH -1 and y == HEIGHT - 1:
        return 100
    if y == HEIGHT - 1 and x > 0 and x < WIDTH - 1:
        return -100
    return -1

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
    done = cx == WIDTH -1 and cy == HEIGHT - 1
    return new_state, r, done

def select_action(Q, s):
    x, y = s
    if random.random() < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)
    return np.argmax(Q[x * HEIGHT + y])

Q_SARAS = np.zeros((WIDTH * HEIGHT, len(ACTIONS)))
HIS_SARAS = np.zeros((HEIGHT, WIDTH))
REWARDS_SARAS = np.array([])

Q= np.zeros((WIDTH * HEIGHT, len(ACTIONS)))
HIS = np.zeros((HEIGHT, WIDTH))
REWARDS = np.array([])
for _ in range(EPISODES):
    total_reward = 0
    done = False
    s = (0, HEIGHT-1)
    a = select_action(Q_SARAS, s)
    while not done:
        x, y = s
        HIS_SARAS[y, x] += 1
        
        _s, r, done = take_action(s, a)

        _a = select_action(Q_SARAS, _s)

        _x, _y = _s
        Q_SARAS[x * HEIGHT + y, a] = Q_SARAS[x * HEIGHT + y, a] + LEARNING_RATE * (r + GAMMA * Q_SARAS[_x * HEIGHT + _y, _a] - Q_SARAS[x * HEIGHT + y, a])
        s = _s
        a = _a
        total_reward += r

    REWARDS_SARAS = np.append(REWARDS_SARAS, total_reward)
    
    total_reward = 0
    done = False
    s = (0, HEIGHT-1)
    while not done:
        x, y = s
        HIS[y, x] += 1
        a = select_action(Q, s)
        _s, r, done = take_action(s, a)

        _x, _y = _s
        Q[x * HEIGHT + y, a] = Q[x * HEIGHT + y, a] + LEARNING_RATE * (r + GAMMA * np.max(Q[_x * HEIGHT + _y]) - Q[x * HEIGHT + y, a])
        s = _s
        total_reward += r
    REWARDS = np.append(REWARDS, total_reward)


mean_SARSA = np.mean(REWARDS_SARAS.reshape(-1, 5), axis=1)
mean_Q = np.mean(REWARDS.reshape(-1, 5), axis=1)

fig, axs = plt.subplots(2)
axs[0].set_title('SARSA')
axs[0].imshow(HIS_SARAS, cmap='viridis', interpolation='nearest')
axs[1].set_title('Q-Learning')
axs[1].imshow(HIS, cmap='viridis', interpolation='nearest')
plt.savefig("positions.png")
fig, axs = plt.subplots(1)
axs.set_title("SARSA - Rewards")
axs.plot(np.arange(np.shape(mean_SARSA)[0]), mean_SARSA, label="SARSA")
axs.plot(np.arange(np.shape(mean_Q)[0]), mean_Q, label="Q-Learning")
axs.legend()
plt.savefig("rewards.png")
