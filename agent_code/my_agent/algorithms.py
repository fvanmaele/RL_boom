import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
import pickle
import copy

from settings import s
from settings import e

############# USEFUL FUNCTIONS ##############

def get_blast_coords(arena, x, y):
    """Retrieve the blast range for a bomb.

    The maximal power of the bomb (maximum range in each direction) is
    imported directly from the game settings. The blast range is
    adjusted according to walls (immutable obstacles) in the game
    arena.

    Parameters:
    * arena:  2-dimensional array describing the game arena.
    * x, y:   Coordinates of the bomb.

    Return Value:
    * Array containing each coordinate of the bomb's blast range.
    """
    bomb_power = s.bomb_power
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


# Function modified from simple_agent to return the path towards a
# target, instead of only the first step.
def look_for_targets_path(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until
    a target is encountered.  If no target can be reached, the path
    that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        the path towards closest target or towards tile closest to any
        target, beginning at the next step.
    """
    if len(targets) == 0:
        return []

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
    if logger:
        logger.debug(f'Suitable target found at {best}')

    # Determine the path towards the best found target tile, start not included
    current = best
    path = []
    while True:
        path.insert(0, current)
        if parent_dict[current] == start:
            return path
        current = parent_dict[current]


def look_for_targets(free_space, start, targets, logger=None):
    """Returns the coordinate of first step towards closest target, or
    towards tile closest to any target.
    """
    path = look_for_targets_path(free_space, start, targets, logger=None)

    if len(path):
        return path[0]
    else:
        return None


############# FEATURES ##############

"""
In all of the next features, the following action order is assumed:
  'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'
"""

def feature1(game_state):
    coins = game_state['coins']
    bombs = game_state['bombs']
    arena = game_state['arena']
    agent = game_state['self']

    # Check the arena (np.array) for free tiles, and include the
    # comparison result as a boolean np.array.
    free_space = arena == 0
    # We do not include agents as obstacles, as they are likely to
    # move in the next round.
    for xb, yb, _ in bombs:
        free_space[xb, yb] = False

    return feature1_base(agent, bombs, coins, free_space)


def feature1_base(agent, bombs, coins, free_space):
    """Reward the agent to move in a direction towards a coin.

    By definition, only move actions can be rewarded by this feature.
    """
    x, y, _, bombs_left = agent
    feature = [] # Desired feature
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    best_direction = look_for_targets(free_space, (x, y), coins)

    # Check if there are coins available in the game.
    if best_direction is None:
        return np.zeros(6)
    for d in directions:
        if d == best_direction:
            feature.append(1)
        else:
            feature.append(0)

    # Only move actions allow the agent to collect a coin.
    feature.append(0) # 'BOMB'
    feature.append(0) # 'WAIT'

    return feature


def feature2(game_state):
    agent = game_state['self']
    bombs = game_state['bombs']
    others = game_state['others']
    explosions = game_state['explosions']
    arena = game_state['arena']

    # The actual (discounted for walls) blast range of a bomb is only
    # available if a bomb has already exploded. We thus compute it
    # manually with 'get_blast_coords'.
    danger_zone = []
    if len(bombs) != 0:
        for xb, yb, _ in bombs:
            danger_zone += get_blast_coords(arena, xb, yb)

    return feature2_base(agent, arena, bombs, others, explosions, danger_zone)


def feature2_base(agent, arena, bombs, others, explosions, danger_zone):
    x, y, _, _ = agent
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others_xy = [(x,y) for (x,y,n,b) in others]
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    feature = []

    # bomb_map gives the maximum blast range of a bomb, if walls are
    # not taken into account. The values in this array are set to the
    # timer for each bomb.
    bomb_map = np.ones(arena.shape) * 5
    for xb, yb, t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)

    for d in directions:
        # Check if the tile reached by the next action is occupied by an
        # object. (Opposing agents may wait, thus we should check them
        # even if they can move away.) This object may be destroyed by bombs,
        # but prevents us from moving into a free tile.
        if ((arena[d] != 0) or
            (d in others_xy) or
            (d in bombs_xy)):
            d = (x, y)

        # We first check if the agent moves into the blast range of a
        # bomb which will explode directly after. The second condition
        # checks if the agent moves into an ongoing explosion. In both
        # cases, such a movement causes certain death for the agent
        # (that is, we set F_i(s, a) = 1).
        if ((d in danger_zone) and bomb_map[d] == 0) or explosions[d] > 1:
            feature.append(1)
        else:
            feature.append(0)

    # In this feature, placing a bomb is equivalent to waiting.
    feature.append(feature[-1])

    return feature


def feature3(game_state):
    agent = game_state['self']
    arena = game_state['arena']
    bombs = game_state['bombs']
    others = game_state['others']

    danger_zone = []
    if len(bombs) != 0:
        for xb, yb, _ in bombs:
            danger_zone += get_blast_coords(arena, xb, yb)

    return feature3_base(agent, arena, bombs, others, danger_zone)


def feature3_base(agent, arena, bombs, others, danger_zone):
    """Penalize the agent for going or remaining into an area threatened
    by a bomb.

    The logic used in this feature is very similar to feature2. The
    main difference is that we consider any bomb present in the arena,
    not only those that will explode in the next step.
    """
    x, y, _, bombs_left = agent
    others_xy = [(x,y) for (x,y,n,b) in others]
    bombs_xy = [(x,y) for (x,y,t) in bombs]

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    feature = []

    for d in directions:
        # Check if the tile reached by the next action is occupied by
        # an object.
        if ((arena[d] != 0) or
            (d in others_xy) or
            (d in bombs_xy)):
            d = (x,y)

        if d in danger_zone:
            feature.append(1)
        else:
            feature.append(0)

    # In this feature, placing a bomb is equivalent to waiting.
    feature.append(feature[-1])

    return feature


def feature4(game_state):
    agent = game_state['self']
    bombs = game_state['bombs']
    arena = game_state['arena']
    others = game_state['others']

    # Compute the range of all bombs in the game and their blast
    # radius ('danger zone')
    danger_zone = []
    for xb, yb, _ in bombs:
        danger_zone += get_blast_coords(arena, xb, yb)

    # The agent can use any free tile in the arena to escape from a
    # bomb (which may not immediately explode).
    free_space = arena == 0
    # We do not include agents as obstacles, as they are likely to
    # move in the next round.
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    return feature4_base(agent, arena, bombs, danger_zone, free_space)


def feature4_base(agent, arena, bombs, danger_zone, free_space):
    """Reward the agent for moving in the shortest direction outside
    the blast range of (all) bombs in the game.
    """
    x, y, _, bombs_left = agent
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    feature = []

    # Check if the arena contains any bombs with a blast radius
    # containing the agent.
    if len(bombs) == 0 or (x, y) not in danger_zone:
        return np.zeros(6)

    # Define a "safe zone" of all free tiles in the arena, which are
    # not part of the above danger zone.
    safe_zone = [(x,y) for x in range(1,16) for y in range(1,16) if
                 (arena[x, y] == 0) and (x,y) not in danger_zone]

    # Check if the agent can move into a safe area.
    if len(safe_zone) == 0:
        return np.zeros(6)
    safety_direction = look_for_targets(free_space, (x, y), safe_zone)

    for d in directions:
        if d == safety_direction:
            feature.append(1)
        else:
            feature.append(0)

    # When the agent is retreating from bombs, we do not wish to
    # expand the danger zone by dropping a bomb ourselves.
    feature.append(feature[-1])
    feature[-2] = 0

    return feature


def feature5(game_state):
    """
    Penalize the agent taking an invalid action.
    """
    agent = game_state['self']
    arena = game_state['arena']
    bombs = game_state['bombs']
    others = game_state['others']

    return feature5_base(agent, arena, bombs, others)


def feature5_base(agent, arena, bombs, others):
    x, y, _, bombs_left = agent
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others_xy = [(x,y) for (x,y,n,b) in others]

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    feature = []

    # Check if the next movement would move the agent into an object.
    # When checking other objects than walls (immutable), we make the
    # following observations regarding invalid actions. Which agents
    # moves first is decided randomly; an initially free square may
    # thus be later occupied by an agent. Crates may be destroyed by a
    # ticking bomb, but this is done only after all agents have
    # performed their respective agents.
    for d in directions:
        if ((arena[d] != 0) or
            (d in others_xy) or
            (d in bombs_xy)):
            feature.append(1)
        else:
            feature.append(0)

    # Attempting to place a bomb with none remaining is invalid.
    if bombs_left == 0:
        feature.append(1)
    else:
        feature.append(0)

    # Remaining in-place is a valid action for the agent.
    feature.append(0)

    return feature


def feature6(game_state):
    """
    Reward the agent collecting a coin.
    """
    agent = game_state['self']
    coins = game_state['coins']

    return feature6_base(agent, coins)


def feature6_base(agent, coins):
    x, y, _, _ = agent
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    feature = []

    for d in directions:
        if d in coins:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)

    return feature


def feature7(game_state):
    """
    Reward putting a bomb next to a crate.  F(s,a) = 0 for all actions
    and F(s,a) = 1 for a 'BOMB' if we are next to a crate.
    """
    agent = game_state['self']
    arena = game_state['arena']

    return feature7_base(agent, arena)


def feature7_base(agent, arena):
    x, y, _, bombs_left = agent
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]

    feature = [0, 0, 0, 0] # Move actions

    # add feature for the action 'BOMB'
    # check if we are next to a crate
    CHECK_FOR_CRATE = False
    for d in directions:
        if arena[d] == 1:
            CHECK_FOR_CRATE = True
            break
    if CHECK_FOR_CRATE and bombs_left > 0:
        feature.append(1)
    else:
        feature.append(0)

    # add feature for 'WAIT'
    feature.append(0)

    return feature


def feature8(game_state):
    """
    Reward (if there are no blocks anymore ? and no coins?)  the
    available movements F(s,a) = 1, otherwise F(s,a) = 0 .  Bombs = 0,
    WAIT =1 ?
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


def feature9(game_state):
    """
    Reward going into dead-ends (from simple agent)
    """
    agent = game_state['self']
    arena = game_state['arena']
    bombs = game_state['bombs']
    others = game_state['others']

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    # TODO: take timer of bomb into account?
    for xb, yb, t in bombs:
        free_space[xb, yb] = False

    return feature9_base(agent, arena, bombs, others, free_space)


def feature9_base(agent, arena, bombs, others, free_space):
    x, y, _, bombs_left = agent
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others_xy = [(x,y) for (x,y,n,b) in others]

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    feature = []

    # Compute dead ends, i.e. tiles with only a single neighboring,
    # free tile. Only crates and walls are taken into account;
    # opposing agents and bombs are ignored: moving the agent towards
    # these targets may impose a negative reward, and should be left
    # to other features.
    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0) and
                 ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
    best_direction = look_for_targets(free_space, (x,y), dead_ends)

    # Do not reward if the agent is already in a dead-end, or if there
    # are none in the arena.
    if (x, y) in dead_ends or best_direction is None:
        return np.zeros(6)

    for d in directions:
        if d != best_direction:
            feature.append(0)
        else:
            feature.append(1)

    # Only a move action can move the agent into a dead end.
    feature.append(0)
    feature.append(0)

    return feature


def feature10(game_state):
    """
    Reward going to crates.
    """
    arena = game_state['arena']
    agent = game_state['self']
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    # We do not include agents as obstacles, as they are likely to
    # move in the next round.
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    return feature10_base(agent, arena, crates, free_space)


def feature10_base(agent, arena, crates, free_space):
    x, y, _, bombs_left = agent
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    feature = []

    # Observation: look_for_targets requires that any targets are
    # considered free space.
    for xc, yc in crates:
        free_space[xc, yc] = True

    # Check if crates are available in the game.
    best_direction = look_for_targets(free_space, (x,y), crates)
    if best_direction is None: # len(crates) == 0
        return np.zeros(6)

    # If we are directly next to a create, look_for_targets will
    # return the tile where the agent is located in, rewarding an
    # (unnecessary) wait action.
    if best_direction == (x,y):
        return np.zeros(6)

    # A-priori, blowing up one crate is not better than blowing up
    # another; therefore, return 0 for every action (move, bomb, wait)
    # if the agent is next to a crate.
    for d in directions:
        if d in crates:
            return np.zeros(6)

    for d in directions:
        if d == best_direction:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)

    return feature


def feature11(game_state):
    """
    Penalize moving getting into dead_ends if the last action was 'BOMB'
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']

    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]

    feature = []  # feature that we want get

    for d in directions:
        if (d in dead_ends) and ((x,y) in bombs_xy):
            feature.append(1)
        else:
            feature.append(0)

    # For 'BOMB' and 'WAIT'
    feature += [0,0]

    return feature


def feature12(game_state):
    """
    HUNTING MODE: Reward puting a bomb next to an agent

    Nur if len(coins) != 0 or len(crates) !=0:
        return np.zeros(6)
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    coins = game_state['coins']

    feature = [0, 0, 0, 0] # Desired feature

    if len(coins) != 0 or len(crates) !=0:
        return np.zeros(6)

    # add feature for the action 'BOMB'
    # check if we are next to a crate
    CHECK_FOR_OTHERS = False
    for d in others:
        if arena[d] == 1:
            CHECK_FOR_OTHERS = True
            break

    if CHECK_FOR_OTHERS and bombs_left>0:
        feature.append(1)
    else:
        feature.append(0)

    # add feature for 'WAIT'
    feature.append(0)

    return np.asarray(feature)


def feature13(game_state):
    """
    Reward putting a bomb that traps another agent if he is dead-ends
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]

    feature = [0, 0, 0, 0] # Desired feature

    dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
                    and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]

    CHECK_CONDITION = False
    for o in others:
        if o in dead_ends:
            CHECK_CONDITION = True
            break

    CHECK_OTHERS = False
    for d in directions:
        if d in others:
            CHECK_OTHERS= True
            break

    if CHECK_CONDITION and CHECK_OTHERS:
        feature.append(1)
    else:
        feature.append(0)

    # For 'WAIT'
    feature.append(0)

    return feature


def feature14(game_state):
    """
    Hunting mode:
    If no crates and no coins reward going towards the nearest agent
    """

    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    coins = game_state['coins']
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]

    if len(coins) != 0 or len(crates) !=0:
        return np.zeros(6)

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    best_coord = look_for_targets(free_space, (x,y), others)

    feature = []

    if best_coord is None:
        return np.zeros(6)

    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)

    feature += [0,0]

    return feature


def feature15(game_state):
    """
    if not in Hunting mode:
    If no crates and no coins penalize going towards the nearest agent
    """

    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    crates = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 1)]
    coins = game_state['coins']
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]

    if len(coins) == 0 and len(crates) ==0:
        return np.zeros(6)

    # construct the free_space Boolean numpy_array
    free_space = arena == 0
    for xb, yb in bombs_xy:
        free_space[xb, yb] = False

    best_coord = look_for_targets(free_space, (x,y), others)

    feature = []

    if best_coord is None:
        return np.zeros(6)

    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)

    feature += [0,0]

    return feature
