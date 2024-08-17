#
# Created on:  Thu Aug 15 2024
# By:  Lukas Mettler (lukas.mettler@student.kit.edu)
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation


#####################################

dims = (100, 100)
frames = 200
interval = 100

prob_lightning=1e-6
prob_planting=1e-4


initial_tree_prob = 0.6
initial_grid = np.random.choice(a=[1, 0], p=[initial_tree_prob, 1-initial_tree_prob], size=dims)

#initial fires
initial_grid[20, 20] = 2
initial_grid[70, 50] = 2

####################################





def neigbors2fire(fire_mask):
    # for each dimension shift this forward/backward one 
    to_fire_mask = np.zeros_like(fire_mask).astype(bool)

    for dim in range(len(fire_mask.shape)):
        tfm = fire_mask.copy()  # booleans of size grid
        for shift_direction, clear_index in zip([-1, 1], [-1, 0]):
            shifted_mask = np.roll(tfm, shift=shift_direction, axis=dim)
            shifted_mask.swapaxes(0, dim)[clear_index] = False

            to_fire_mask = np.logical_or(to_fire_mask, shifted_mask)

    return to_fire_mask
        


def step(grid, prob_lightning=0.02, prob_planting=0.02):
    elements = grid.copy()
    
    empty_mask = grid == 0
    tree_mask = grid == 1
    fire_mask = grid == 2

    # tree turns to fire if neighbor(s) are fire
    next_to_fire_mask = neigbors2fire(fire_mask)
    trees_next_to_fire_mask = np.logical_and(tree_mask, next_to_fire_mask)
    elements[trees_next_to_fire_mask] = 2
    
    # previous fire turns to empty
    elements[fire_mask] = 0

    # empty turns to tree with prob.
    elements[empty_mask] = np.random.choice(a=[1, 0], p=[prob_planting, 1-prob_planting], size=elements[empty_mask].shape)

    # tree turns to fire with prob.
    tree_mask = np.logical_and(tree_mask, ~trees_next_to_fire_mask)
    elements[tree_mask] = np.random.choice(a=[2, 1], p=[prob_lightning, 1-prob_lightning], size=elements[tree_mask].shape)

    return elements



def analyzeforest(grid):
    n_tot = np.array(grid.shape).prod()
    n_empty=  100. * (grid== 0).astype(float).flatten().sum() / n_tot
    n_tree =  100. * (grid== 1).astype(float).flatten().sum() / n_tot
    n_fire =  100. * (grid== 2).astype(float).flatten().sum() / n_tot
    
    return {'empty': n_empty, 'fire': n_fire, 'tree': n_tree}


######################################################################################
######################################################################################
######################################################################################

current_grid = initial_grid.copy()

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8,10), gridspec_kw={'height_ratios': [2, 1]}, tight_layout=True)

colors = ['gray', 'green', 'red']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(range(0,4), cmap.N)


container = ax0.imshow(current_grid, animated=True, cmap=cmap, norm=norm)
ax0.set_xticks([])
ax0.set_yticks([])


state_dict = analyzeforest(current_grid)
nums = [list(state_dict.values())]
labels = list(state_dict.keys())
colors_stack = ['gray', 'red', 'green']
plot = ax1.stackplot([0], np.transpose(nums), colors=colors_stack, labels=labels)
ax1.set_xlim(0, frames)
ax1.set_ylim(0, 100.)
ax1.set_xlabel('step')
ax1.set_ylabel('% of forest')
ax1.legend(loc='lower center', bbox_to_anchor=(0.8, 1.05), ncol=3)

text_position = (10, 90)
text = ax1.text(*text_position, f'current step: {0}')

timeline = [current_grid]
artists = [[container, *plot, text]]

for i in range(1, frames):
    current_grid = step(current_grid, prob_lightning=prob_lightning, prob_planting=prob_planting)
    state_dict = analyzeforest(current_grid)
    nums.append(list(state_dict.values()))

    # plots
    container = ax0.imshow(current_grid, animated=True, cmap=cmap, norm=norm)
    plot = ax1.stackplot(range(i+1), np.transpose(nums), colors=colors_stack)
    text = ax1.text(*text_position, f'current step: {i}')
    artists.append([container, *plot, text])


print('All artists calculated!')
anim = animation.ArtistAnimation(fig=fig, artists=artists, interval=interval, blit=True)
#anim.save('file.mp4')
plt.show()
