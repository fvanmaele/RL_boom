import copy
import numpy as np
from settings import s

## Required algorithms for feature extraction.

def taxi_cab_metric(p, q):
    """Definition of the Manhattan (or taxi-cab) metric.

    The taxicab metric d between two vectors p, q in an n-dimensional
    real vector space with fixed Cartesian coordinate system, is the
    sum of the lengths of the projections of the line segment between
    the points onto the coordinate axes.

    Note that this metric does not account for non-reachable tiles.

    Parameters:
    * p: coordinate vector of the first point
    * q: coordinate vector of the second point

    Return Value:
    * d(p, q)
    """
    return np.sum(np.absolute(np.subtract(p, q)))


# Function adapted from items.py.
def get_blast_coords(x, y, arena):
    """Retrieve the blast range for a bomb.

    The maximal power of the bomb (maximum range in each direction) is
    imported directly from the game settings. The blast range is
    adjusted according to walls (immutable obstacles) in the game
    arena.

    Parameters:
    * x, y:  Coordinates of the bomb.
    * arena: 2-dimensional array describing the game arena.

    Return Value:
    * Array containing each coordinate of the bomb's blast range.
    """
    blast_coords = [(x,y)]
    power = s.bomb_power
    #assert(power == '3')

    for i in range(1, power+1):
        if arena[x+i, y] == -1:
            break
        blast_coords.append((x+i, y))
    for i in range(1, power+1):
        if arena[x-i, y] == -1:
            break
        blast_coords.append((x-i, y))
    for i in range(1, power+1):
        if arena[x, y+i] == -1:
            break
        blast_coords.append((x, y+i))
    for i in range(1, power+1):
        if arena[x, y-i] == -1:
            break
        blast_coords.append((x, y-i))

    return blast_coords


# Function adapted from environment.py
# TODO: This does not include the location of the agent!
def tile_is_free(x, y, arena, bombs, others):
    """Check if a tile is occupied by an obstacle.

    An obstacle is any wall, crate, bomb, or opposing agent.
    
    Parameters:
    * x, y:    coordinates (x,y) of the tile to check.
    * arena:   2-dimensional np.array containing markers for walls (-1)
               and crates (1) in the arena.
    * bombs:   Array of tuples containing bomb coordinates.
    * others:  Array of tuples containing opposing agent coordinates.

    Return Value:
    * is_free: True if (x, y) is a free tile, False if not.
    """
    is_free = (arena[x, y] == 0)

    if is_free:
        for obstacle in bombs + others:
            is_free = is_free and (obstacle[0] != x or obstacle[1] != y)

    # Return combined value.
    return is_free


# Code from simple_agent for path finding.
def look_for_targets(free_space, start, targets, return_distance=False, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until
    a target is encountered.  If no target can be reached, the path
    that takes the agent closest to any target is chosen.

    Arguments:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start:      Coordinate from which to begin the search.
        targets:    List or array holding the coordinates of all target tiles.
        logger:     Optional logger object for debugging.

    Returns:
        Coordinate of first step towards closest target or towards
        tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    distance_all = 0
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
                distance_all += 1
                dist_so_far[neighbor] = dist_so_far[current] + 1

    # Write debug output
    if logger:
        logger.debug(f'Suitable target found at {best}')

    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            if return_distance:
                return current, distance_all
            else:
                return current
        current = parent_dict[current]



## Feature extraction F(S, A) used in Q-learning

# TODO: rename functions after their purpose, then stack with
# feature_1(foo, ...)

## Coin features

# def feature_1(arena, coins, x, y, x_n, y_n):

# TODO: stack for coins, crates, and dead ends (?)
# dead_ends = [(x,y) for x in range(1,16) for y in range(1,16) if (arena[x,y] == 0)
# and ([arena[x+1,y], arena[x-1,y], arena[x,y+1], arena[x,y-1]].count(0) == 1)]
def reward_target(arena, targets, x, y, x_n, y_n):
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


def reward_coin_collect(coins, x_n, y_n):
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
def penalize_coin_distance(coins, x_n, y_n):
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

def reward_targets_radius(targets, x_n, y_n, radius=3):
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

def penalize_obstacle(arena, bombs, others, x, y, x_n, y_n):
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
def reward_bomb_crate_greedy(x, y, bombs_left, arena, action):
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


def reward_bomb_crate(x, y, bombs_left, arena, action):
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

def penalize_agent_death(x, y, x_n, y_n, arena, bombs, explosions, others):
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


def feature_4(state):
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
    for x_b, y_b in bombs_xy:
        danger_zone += get_blast_coords(x_b, y_b, arena)
        
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


# TODO: Beschreibung
def reward_bomb_escape(x, y, x_n, y_n, arena, bombs):
    """Minimize the distance to safety should the agent be in a zone where
    an explosion will occur.
    
    Parameters:
    *

    Return value:
    * 1:  the action a reduces distance to safety
    * 0:  otherwise.

    """
    # If there are no bombs in the arena, there is no safe area to
    # escape to by definition.
    if len(bombs) == 0:
        return 0

    danger_zone = []
    for x_b, y_b, t in bombs:
        # Append new blast coordinates to "danger zone".
        danger_zone += get_blast_coords(x_b, y_b, arena):

    # If the agent is outside the danger zone (blast range of any
    # bomb), it is already in saftey.
    if (x, y
    return 0

    # The agent may use tiles inside the blast range to navigate away
    # from the 'danger zone', as the bomb may not immediately explode
    # (e.g, when placed by the own agent).
    free_tiles = (arena == 0) # boolean np.ndarray

        """
        We then check if the agent is in the danger zone
        If agent is not in the danger zone, we return 0.
        Otherwise we calculate the distance/direction of safety.
        """
        if agent[0] in danger_zone[:,0] and agent[1] in danger_zone[:,1]:            
            """
            We then mark these explosions on our map. here we deep-copy
            the arena, so that in the case that the arena is needed for
            other features, it remains unchanged
            """
            map_ = copy.deepcopy(arena)
            map_[danger_zone[:,0], danger_zone[:,1]] = 2

            safe_loc = np.argwhere(map_==0)
            free_space = abs(map_) != 1
            d = look_for_targets(free_space, (agent[0], agent[1]), safe_loc)
            print(d)

            """
            We then calculate the minimum distance of our agent to any
            of these safe locations.  For simplicity, only Manhattan
            distance is used to calculate distance in this feature
            extraction. However, this doesn't always represent the
            true distance in the game because of the walls.(Possible
            point of improvement?)
 
            Then, we calculate the positions of our agent after taking
            all possible actions.
            """
            
            actions_loc = np.array([(agent[0], agent[1]-1), #up
                                    (agent[0], agent[1]+1), #down
                                    (agent[0]-1, agent[1]), #left
                                    (agent[0]+1, agent[1]), #right
                                    (agent[0], agent[1]),   #bomb
                                    (agent[0], agent[1])])  #wait
            
            res = (actions_loc[:,0] == d[0]) & (actions_loc[:,1] == d[1])
            res = res.astype(int)
            
        return res


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

# TODO: Add a feature for the remaining time?
# no, not dependent on action
