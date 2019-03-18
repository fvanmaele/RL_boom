import numpy as np
from agent_code.my_agent.algorithm.arena import *


## Feature extraction F(S, A) used in Q-learning

# TODO: rename functions after their purpose, then stack with
# feature_1(foo, ...)

## Coin features

# def feature_1(arena, coins, x, y, x_n, y_n):

# TODO: stack for coins, crates, and dead ends (?)
# dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
# and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
def feature_1(arena, targets, x, y, x_n, y_n):
    """Reward moving in a direction towards a target (coin, crate, or dead end).

    Parameters:
    * arena:     np.array for the game arena in state s.
    * targets:   list with target objects in state s.
    * x, y:      coordinates of the agent in state s.
    * x_n, y_n:  coordinates of the agent after action a.

    Return Value:
    * 1:  action is the first step in reaching a coin in the arena.
    * 0:  otherwise.
    """
    # Only move actions allow the agent to pick up a coin.
    if x == x_n and y == y_n:
        return 0

    # Check the arena (np.array) for free tiles, and include the
    # comparison result as a boolean np.array.
    free_space = (arena == 0)
    best_coord = look_for_targets(free_space, (x, y), targets)

    # Check if any coins were found. If so, check if the next action
    # will move the agent towards a coin.
    if best_coord is None:
        return 0 # TODO: When precisely is best_coord 'None'?
    else:
        return (x_n, y_n) == best_coord


def feature_2(coins, x_n, y_n):
    """Reward collecting a coin.

    Parameters:
    * coins:     list with coin coordinates in state s.
    * x_n, y_n:  coordinates of the agent after action a.

    Return Value:
    * 1: action most likely causes the agent to collect a coin.
    * 0: otherwise.
    """
    # Only move actions (UP, DOWN, LEFT, RIGHT) allow the agent to
    # collect a coin. A tile cannot be occupied by both an obstacle
    # and a coin; it thus suffices to compare the coin coordinates,
    # and not explicitely check for free tiles.
    for coin in coins:
        if x_n == coin[0] and y_n == coin[1]:
            return 1
    # No coin was collected.
    return 0


# TODO: Use breadth-first search?
# TODO: Use for general targets (coins, crates). Note that being close
# to opposing agents may have an adverse effect!
def feature_3(coins, x_n, y_n):
    """Penalize agent distance to coins.

    Parameters:
    * coins:     list of Coin objects in state s.
    * x_n, y_n:  coordinates of the agent after action a.

    Return Value:
    * d:  distance between our agent and the nearest coin. If no coin
          was found, return 30.
    """
    min_distance = 30 # 15x15 arena

    for coin in coins:
        d = taxi_cab_metric((x_n, y_n), (coin[0], coin[0]))
        if d < min_distance:
            min_distance = d

    return min_distance


# TODO: Stack feature with different radii, e.g. 3, 5, 10, 12.
# TODO: Combine all distances in one go (duplicate for different radii
# and feature_3)

# def feature_4(coins, x_n, y_n, radius=3):
# def feature_4(coins, x_n, y_n, radius=5):
# def feature_4(coins, x_n, y_n, radius=10):
# def feature_4(coins, x_n, y_n, radius=12):

def feature_4(targets, x_n, y_n, radius=3):
    """Reward an amount of targets in a certain radius around the agent.

    Parameters:
    * targets:   list of target objects in state s.
    * x_n, y_n:  coordinates of the agent after action a.
    * radius:    target radius, defaults to 3.

    Return Value:
    * Amount of targets in a ball around the agent.
    """
    target_distance = radius
    counter = 0

    for target in targets:
        d = taxi_cab_metric((x_n, y_n), (target[0], target[1]))
        if d <= target_distance:
            counter += 1

    return counter


## Movement features

def feature_5(arena, bombs, others, x, y, x_n, y_n):
    """Penalize the agent moving into an obstacle. (invalid action)

    Parameters:
    * arena:     np.array for the game arena in state s.
    * bombs:     list of tuples representing bombs in state s.
    * others:    list of opposing agents in state s.
    * x, y:      coordinates of the agent in state s.
    * x_n, y_n:  coordinates of the agent after action a.
    
    Return Value:
    * 1:  action most likely takes the agent into an obstacle.
    * 0:  otherwise.
    """
    # Do not penalize placing a bomb or waiting, as these are
    # valid actions.
    if x == x_n and y == y_n:
        return 0
    elif not tile_is_free(x_n, y_n, arena, bombs, others):
        return 1    
    else:
        return 0 # No obstacle found


## Crate features

# def feature_1(arena, crates, x, y, x_n, y_n):


# def feature_4(targets, x_n, y_n, radius=3):
# def feature_4(targets, x_n, y_n, radius=5):
# def feature_4(targets, x_n, y_n, radius=10):
# def feature_4(targets, x_n, y_n, radius=12):

# Note: only feature where 'action' is specified explicitely
# (distinction between 'WAIT' and 'BOMB')
def feature_6(x, y, bombs_left, arena, action):
    """Reward placing bombs directly next to a crate.

    Parameters:
    * x, y:        coordinates of the agent in state s.
    * bombs_left:  amount of remaining bombs for the agent in state s.
    * arena:       np.array for the game arena in state s.
    * action:      action transitioning the agent from state s to state s'.

    Return Value:
    * 1:  agent will place a bomb next to a create.
    * 0:  otherwise.
    """
    # Only the BOMB action allows the agent to destroy a coin.
    if action == 'BOMB' and bombs_left > 0:
        # Check if the agent is in the immediate vicinity of a crate.
        if (arena[x+1, y] == 1 or arena[x-1, y] == 1 or
            arena[x, y+1] == 1 or arena[x, y-1] == 1):
            # TODO: We can't move into a square with an existing bomb,
            # so this action should not be possible.

            # for xb, yb, t in bombs:
            #     if x == xb and y == yb:
            #         return 0
            # No bomb is placed in the agent's current position.
            return 1

    # The agent is far removed from a crate, makes a movement, or has
    # no bombs remaining.
    return 0


def feature_7(x, y, bombs_left, arena, action):
    """Reward placing bombs at long distance from a crate.
    
    Parameters:
    * x, y:        coordinates of the agent in state s.
    * bombs_left:  amount of remaining bombs for the agent in state s.
    * arena:       np.array for the game arena in state s.
    * action:      action transitioning the agent from state s to state s'.

    Return Value:
    * n:  amount of crates destroyed by an agent placing a bomb.
    * 0:  otherwise.
    """
    crates_destroyed = 0

    if action == 'BOMB' and bombs_left > 0:
        # Check if the blast range of a dropped bomb can reach a crate.
        for x, y in get_blast_coords(x, y, arena):
            if arena[(x, y)] == 1:
                crates_destroyed += 1

    return crates_destroyed

## Bomb features

def feature_8(x, y, x_n, y_n, arena, bombs, explosions, others):
    """Penalize taking an action causing the agent to die.

    Parameters:
    * x, y:        coordinates of the agent in state s.
    * x_n, y_n:    coordinates of the agent in next state s'.
    * arena:       np.array for the game arena in state s.
    * bombs:       list of tuples representing bombs in state s.
    * explosions:  np.array representing ongoing explosions in state s.
    * others:      coordinates of opposing agents in state s.

    Return Value:
    * 1:  action causes the agent to die on action a.
    * 0:  otherwise.
    """
    target = (x_n, y_n)
    # Check if the tile reached by the next action is occupied by an
    # object. (Opposing agents may wait, thus we should check them
    # even if they can move away.) This object may be destroyed by bombs,
    # but prevents us from moving into a free tile.
    if not tile_is_free(target[0], target[1], arena, bombs, others):
        # Set the next action as equivalent to waiting. (Such an
        # action is invalid, as handled by a different feature.)
        target = (x, y)

    # Check if the agent will move into the blast range of an existing
    # bomb. This information is only available when the bomb has
    # already exploded, and has to be calculated manually when not.
    if explosions[target] > 1:
        return 1

    # Check if the agent will move into the blast range of a bomb that
    # explodes on the next time step. As the 'explosions' array does
    # not yet contain the blast range, we compute it manually by
    # iterating over each bomb.
    for xb, yb, t in bombs:
        if timer < 0 and target in get_blast_coords(xb, yb, arena):
            return 1

    # If the above conditions are not satisfied, the agent is most
    # likely safe from death by taking the given action.
    return 0

# TODO: Put a limit on how many bombs to use for blowing up crates?

## Agent features

# TODO: place a bomb when directly next to another agent
# Note: similar to feature_6

# TODO: "hunting mode" to add other agents as targets (e.g. if no
# crates or coins are left)

# TODO: create an "unsafe zone" for nearby agents? (for those agents
# that do not evade properly)

## Other features

# TODO: What to do when reaching a dead end? (cf. simple agent)
