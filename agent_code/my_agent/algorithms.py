import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
import pickle
import copy

from settings import s
from settings import e


############# USEFUL FUNCTIONS ##############

def get_blast_coords(arena, bomb):
    """Retrieve the blast range for a bomb.

    The maximal power of the bomb (maximum range in each direction) is
    imported directly from the game settings. The blast range is
    adjusted according to walls (immutable obstacles) in the game
    arena.

    Parameters:
    * arena:  2-dimensional array describing the game arena.
    * bomb:   Coordinates of the bomb.

    Return Value:
    * Array containing each coordinate of the bomb's blast range.
    """
    bomb_power = s.bomb_power
    x, y = bomb[0], bomb[1] 
    blast_coords = [(x,y)]

    for i in range(1, bomb_power+1):
        if arena[x+i, y] == -1: break
        blast_coords.append((x+i,y))
    for i in range(1, bomb_power+1):
        if arena[x-i, y] == -1: break 
        blast_coords.append((x-i, y))
    for i in range(1, bomb_power+1):
        if arena[x, y+i] == -1: break 
        blast_coords.append((x, y+i))
    for i in range(1, bomb_power+1):
        if arena[x, y-i] == -1: break 
        blast_coords.append((x, y-i))

    return blast_coords


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.

    USEFUL FOR feature1
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def get_blast_coords(bomb, arena, arr):
    x, y = bomb[0], bomb[1]
    if len(arr)== 0:
       arr = [(x,y)]
       #np.append(a, [[0,1]], axis=0)
    
    for i in range(1, 3+1):
        if arena[x+i,y] == -1: break
        arr.append((x+i,y))
    for i in range(1, 3+1):
        if arena[x-i,y] == -1: break
        arr.append((x-i,y))           
    for i in range(1, 3+1):
        if arena[x,y+i] == -1: break
        arr.append((x,y+i))            
    for i in range(1, 3+1):
        if arena[x,y-i] == -1: break
        arr.append((x,y-i))
    return arr

############# FEATURES ##############

def feature1(game_state):
    """
    Reward the best possible action to a coin, if it is reachable F(s,a)=1,  otherwise F(s,a)=0.    
    `BOMB' and `WAIT ' are always 0
    """
    coins = game_state['coins']
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']

    feature = [] # Desired feature
    
    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    best_coord = look_for_targets(free_space, (x,y), coins)

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]

    if best_coord is None:
        return np.zeros(6)
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)
    
   # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)
    
    return np.asarray(feature)


def feature2(game_state):
    """
    Penalize if the action follows the agent to death (F,s)=1, F(s,a)=0. otherwise.
    it could be maybe get better
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    explosions = game_state['explosions'] 
    arena = game_state['arena']
    bomb_power = s.bomb_power
    blast_coords = bombs_xy 
    feature = []
    
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    danger_zone = [] 
    if len(bombs) != 0:
        '''
        If there are bombs on the game board, we map all the explosions
        that will be caused by the bombs. For this, we use the help function
        get_blast_coords. This is an adjusted version of the function with the
        same name from item.py of the original framework
        '''
        for b in bombs_xy:
            danger_zone += get_blast_coords(arena, b)

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            d = (x,y)
            
        if ((d in danger_zone) and  bomb_map[d] == 0) or explosions[d] >1:
            feature.append(1) 
        else:
            feature.append(0)
    
    # BOMB actions should be same as WAIT action
    feature.append(feature[-1])

    return np.asarray(feature)


def feature3(game_state):
    """
    Penalize the agent for going into an area threatened by a bomb.
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    arena = game_state['arena']
    feature = []
    danger_zone = []

    if len(bombs) != 0:
        '''
        If there are bombs on the game board, we map all the explosions
        that will be caused by the bombs. For this, we use the help function
        get_blast_coords. This is an adjusted version of the function with the
        same name from item.py of the original framework
        '''
        for b in bombs_xy:
            danger_zone += get_blast_coords(arena, b)

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            d = (x,y)

        if d in danger_zone:
            feature.append(1) 
        else:
            feature.append(0)
    
    # BOMB actions should be same as WAIT action
    feature.append(feature[-1])

    return np.asarray(feature)


def feature4(state):
    """Reward the agent for moving in the shortest direction outside
    the blast range of (all) bombs in the game.
    """
    bombs = state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    arena = state['arena']
    others = state['others']    
    x, y, _, _ = state['self']

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    feature = []


    # Compute the blast range of all bombs in the game ('danger zone')
    danger_zone = []
    for b in bombs_xy:
        danger_zone += get_blast_coords(arena, b)

    if len(bombs) == 0 or (x,y) not in danger_zone:
        return np.zeros(6)

    # The agent can use any free tile in the arena to escape from a
    # bomb (which may not immediately explode).
    free_tiles = arena == 0 # boolean np.ndarray

    # the 'safe zone' is the complement of the free tiles and the
    # 'danger zone'. Use deepcopy to preserve free_tiles.
    targets = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x, y] == 0) and (x,y) not in danger_zone]

    # def look_for_targets(free_space, start, targets, logger=None):
    safety_direction = look_for_targets(free_tiles, (x, y), targets)

    # check if next action moves agent towards safety
    for d in directions:
        if d == safety_direction:
            feature.append(1)
        else:
            feature.append(0)

    # Do not reward placing a bomb at this stage.
    feature.append(feature[-1])
    feature[-2] = 0

    return feature

def feature5(game_state):
    """
    Penalize invalid actions.  F(s,a) = 1, otherwise F(s,a) = 0.  
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']

    feature = [] # Desired feature

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB'
    if bombs_left == 0:
        feature.append(1)
    else:
        feature.append(0)

    # for 'WAIT'
    feature.append(0)

    return np.asarray(feature)

def feature6(game_state):
    """
    Reward when getting a coin F(s,a) = 1, otherwise F(s,a) = 0
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    coins = game_state['coins']
    arena = game_state['arena']

    feature = [] # Desired feature

    for d in directions:
        if (x,y) in coins:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)

    return np.asarray(feature)

def feature7(game_state):
    """
    Reward putting a bomb next to a block. 
    F(s,a) = 0 for all actions and F(s,a) = 1 for a 'BOMB' if we are next to a block.
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    
    feature = [0, 0, 0, 0] # Desired feature

    # add feature for the action 'BOMB'
    # check if we are next to a crate
    CHECK_FOR_CRATE = False 
    for d in directions:
        if arena[d] == 1:
            CHECK_FOR_CRATE = True 
            break
    if CHECK_FOR_CRATE:
        feature.append(1)
    else:
        feature.append(0)

    # add feature for 'WAIT'
    feature.append(0)

    return np.asarray(feature)


def feature8(game_state):
    """
    Reward (if there are no blocks anymore ? and no coins?)  the available movements F(s,a) = 1, 
    otherwise F(s,a) = 0 .   Bombs = 0, WAIT =1 ? 
    """

    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    
    feature = [] # Desired feature

    for d in directions:
        if arena[d] == 0:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(1)

    return np.asarray(feature)

def feature9(game_state):
    """
    Reward going into dead-ends (from simple agent)
    """

    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']

    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    best_coord = look_for_targets(free_space, (x,y), dead_ends)

    feature = []
    if best_coord is None:
        return np.zeros(6)
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)
    
    # for 'BOMB' and 'WAIT'
    feature.append(feature[-1])
    feature[-2] = 0

    return feature



def feature10(game_state):
    """
    Reward going to crates 
    """

    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    best_coord = look_for_targets(free_space, (x,y), crates)

    feature = []
    if best_coord is None:
        return np.zeros(6)
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)
    
    # for 'BOMB' and 'WAIT'
    feature.append(feature[-1])
    feature[-2] = 0

    return feature
