#
# Created on: Thu Aug 15 2024
# By: Lukas Mettler (https://github.com/LEMettler)
#
# Docstrings generated with claude.ai
#

import time, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import animation



######################################################################################

# global parameters

dims = (100, 250)
frames = 200
interval = 70

prob_lightning = 1e-3
prob_planting = 8e-2

initial_tree_prob = 0.0

store_file = False
file_name = 'model.gif'

custom_seed = False
seed = 137
######################################################################################

# functions

def get_input(prompt, expected_type, default_value):
    """
    Prompt the user for input with a default value and type checking.

    Args:
        prompt (str): The prompt to display to the user.
        expected_type (type): The expected type of the input.
        default_value: The default value to use if no input is provided.

    Returns:
        The user input converted to the expected type, or the default value.
    """
    while True:
        user_input = input(f"{prompt} (Enter: {default_value}): ")
        if not user_input:
            return default_value
        try:
            return expected_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a value of type {expected_type.__name__}.")


def promptUser():
    """
    Prompt the user for simulation parameters and update global variables.
    """
    global frames, interval, prob_lightning, prob_planting, initial_tree_prob, store_file, file_name, custom_seed, seed

    if get_input('Custom parameters? (0/1)', int, 0):
            
        frames = get_input("Number of frames", int, frames)
        interval = get_input("Frame interval", int, interval)
        prob_lightning = get_input("Probability of lightning", float, prob_lightning)
        prob_planting = get_input("Enter the probability of planting", float, prob_planting)
        initial_tree_prob = get_input("Initial tree probability", float, initial_tree_prob)
        store_file = bool(get_input('Store output? (0/1)', int, store_file))
        if store_file:
            file_name = get_input("File name", str, file_name)
        custom_seed = bool(get_input('Custom seed? (0/1)', int, custom_seed))
        if custom_seed:
            seed = get_input("Custom seed", int, seed)
            
    printState()


def printState():
    """
    Print the current state of the simulation parameters.
    """
    print(f"Frames: {frames}")
    print(f"Interval: {interval}")
    print(f"Probability of Lightning: {prob_lightning}")
    print(f"Probability of Planting: {prob_planting}")
    print(f"Initial Tree Probability: {initial_tree_prob}")
    if store_file:
        print(f"File Name: {file_name}")
    print(f"Seed: {seed}")
    print('-----------------------------------------')
    


def initializeGrid():
    """
    Initialize the grid with trees and initial fires.

    Returns:
        numpy.ndarray: The initialized grid.
    """
    initial_grid = np.random.choice(a=[1, 0], p=[initial_tree_prob, 1-initial_tree_prob], size=dims)
    return initial_grid


def neigbors2fire(fire_mask):
    """
    Create a mask of cells neighboring fires.

    Args:
        fire_mask (numpy.ndarray): A boolean mask of cells currently on fire.

    Returns:
        numpy.ndarray: A boolean mask of cells neighboring fires.
    """
    to_fire_mask = np.zeros_like(fire_mask).astype(bool)

    for dim in range(len(fire_mask.shape)):
        tfm = fire_mask.copy()
        for shift_direction, clear_index in zip([-1, 1], [-1, 0]):
            shifted_mask = np.roll(tfm, shift=shift_direction, axis=dim)
            shifted_mask.swapaxes(0, dim)[clear_index] = False

            to_fire_mask = np.logical_or(to_fire_mask, shifted_mask)

    return to_fire_mask


def step(grid, prob_lightning=0.02, prob_planting=0.02):
    """
    Perform one step of the forest fire simulation.

    Args:
        grid (numpy.ndarray): The current state of the grid.
        prob_lightning (float): Probability of lightning striking a tree.
        prob_planting (float): Probability of a new tree growing in an empty cell.

    Returns:
        numpy.ndarray: The updated grid after one simulation step.
    """
    elements = grid.copy()
    
    empty_mask = grid == 0
    tree_mask = grid == 1
    fire_mask = grid == 2

    next_to_fire_mask = neigbors2fire(fire_mask)
    trees_next_to_fire_mask = np.logical_and(tree_mask, next_to_fire_mask)
    elements[trees_next_to_fire_mask] = 2
    
    elements[fire_mask] = 0

    elements[empty_mask] = np.random.choice(a=[1, 0], p=[prob_planting, 1-prob_planting], size=elements[empty_mask].shape)

    tree_mask = np.logical_and(tree_mask, ~trees_next_to_fire_mask)
    elements[tree_mask] = np.random.choice(a=[2, 1], p=[prob_lightning, 1-prob_lightning], size=elements[tree_mask].shape)

    return elements


def analyzeForest(grid):
    """
    Analyze the current state of the forest.

    Args:
        grid (numpy.ndarray): The current state of the grid.

    Returns:
        dict: Percentages of empty cells, trees, and fires in the forest.
    """
    n_tot = np.array(grid.shape).prod()
    n_empty = 100. * (grid == 0).astype(float).flatten().sum() / n_tot
    n_tree = 100. * (grid == 1).astype(float).flatten().sum() / n_tot
    n_fire = 100. * (grid == 2).astype(float).flatten().sum() / n_tot
    
    return {'empty': n_empty, 'fire': n_fire, 'tree': n_tree}


def histogramDimensions(grid):
    """
    Calculate histograms for each dimension of the grid.

    Args:
        grid (numpy.ndarray): The current state of the grid.

    Returns:
        list: List of counts for each cell type along each dimension.
    """
    counts = []
    for dim in range(grid.ndim):
        dim_counts = []
        for celltype in [0, 1, 2]:
            count = (grid == 100. * celltype).astype(float).sum(axis=dim) / grid.shape[dim]
            dim_counts.append(count)
        counts.append(dim_counts)
    return counts



######################################################################################


# Get parameters of system and initialize grid
promptUser()

if custom_seed:
    np.random.seed(seed)

initial_grid = initializeGrid()
current_grid = initial_grid.copy()
timeline = [current_grid]


######################################################################################
# stylistic parameters

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10,8), gridspec_kw={'height_ratios': [2, 1]}, tight_layout=True)

colors = ['dimgray', 'darkgreen', 'red']
colors_stack = ['dimgray', 'red', 'darkgreen']

cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(range(0,4), cmap.N)

text_position = (10, 90)

ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title(f'$dim={dims};\; P(\\text{{grow}})={prob_planting};\; P(\\text{{lightning}})={prob_lightning}$')

state_dict = analyzeForest(current_grid)
nums = [list(state_dict.values())]
labels = list(state_dict.keys())
ax1.set_xlim(0, frames)
ax1.set_ylim(0, 100.)
ax1.set_xlabel('step')
ax1.set_ylabel('% of forest')

######################################################################################

# artists and list of artists
container = ax0.imshow(current_grid, animated=True, cmap=cmap, norm=norm)
plot = ax1.stackplot([0], np.transpose(nums), colors=colors_stack, labels=labels)
text = ax1.text(*text_position, f'current step: {0}') # current step
ax1.legend(loc='lower center', bbox_to_anchor=(0.8, 1.05), ncol=3) #legend must be down here

artists = [[container, *plot, text]]


# Main loop
for i in range(1, frames):
    # state progression
    current_grid = step(current_grid, prob_lightning=prob_lightning, prob_planting=prob_planting)
    state_dict = analyzeForest(current_grid)
    nums.append(list(state_dict.values()))

    # artists
    container = ax0.imshow(current_grid, animated=True, cmap=cmap, norm=norm)
    plot = ax1.stackplot(range(i+1), np.transpose(nums), colors=colors_stack)
    text = ax1.text(*text_position, f'current step: {i}')
    artists.append([container, *plot, text])

    # loading bar
    percent = (i + 1) / (frames)
    bar = '#' * int(percent * 50) + '-' * (50 - int(percent * 50))
    sys.stdout.write(f'\r|{bar}| {int(percent * 100)}% ')
    sys.stdout.flush()


# matplotlib animation, storage and display
anim = animation.ArtistAnimation(fig=fig, artists=artists, interval=interval, blit=True)
if store_file:
    anim.save(file_name)
plt.show()
