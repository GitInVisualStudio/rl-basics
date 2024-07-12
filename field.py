from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import random
import matplotlib.patches as mpatches

WIDTH = 12
HEIGHT = 4
EPISODES = 100
EPSILON = 0.1
LEARNING_RATE = 0.1
GAMMA = 0.9
ACTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]



data = [[0] * WIDTH for _ in range(HEIGHT)]
data[-1] = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]
data = np.array(data)

cmap = colors.ListedColormap(['white', 'red', 'blue', 'green'])

# get the unique values from data
# i.e. a sorted list of all values in data
values = np.unique(data.ravel())

fig = plt.figure(figsize=(WIDTH, HEIGHT), frameon=False)
im = plt.imshow(data, cmap=cmap, interpolation="none")

# get the colors of the values, according to the 
# colormap used by imshow
colors = [ cmap(value) for value in values]
# create a patch (proxy artist) for every color 
patches = [ 
    mpatches.Patch(color=colors[0], label="Feld" ),
    mpatches.Patch(color=colors[1], label="Hinderniss" ),
    mpatches.Patch(color=colors[2], label="Agent" ),
    mpatches.Patch(color=colors[3], label="Ziel" ),
]
# put those patched as legend-handles into the legend
plt.legend(handles=patches)
ax = fig.gca()
ax.set_yticks(np.arange(0.5, HEIGHT, 1))
ax.set_xticks(np.arange(0.5, WIDTH, 1))
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)
plt.grid(color='k', linestyle='-', linewidth=1)
plt.savefig("field.png")