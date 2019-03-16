import settings as game_settings

# Save all actions in the action space in a global variable for
# simplicity.
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']


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
    power = game_settings.settings['bomb_power']

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
def tile_is_free(x, y, arena, bombs, others):
    is_free = (arena[x, y] == 0)

    if is_free:
        for obstacle in bombs + others:
            is_free = is_free and (obstacle.x != x or obstacle.y != y)
    return is_free


def vectorized_feature(feature, state):
    """Return a vector containg a feature for each available action.

    Parameters:
    * feature: Callable representing a feature vector F(s, a).
    * state:   Array representing the state s.
    """
    results = []

    for a in actions:
        results.append(feature(state, action))
    return results


def feature_1(state, action):
    """Feature 1

    Value 1: Action is the first step in reaching a coin in the arena.
    Value 0: Otherwise.

    While dropping a bomb can expose a coin, only move actions (up,
    left, down, right) allow the agent to reach it.
    """
    pass


def feature_2(state, action):
    """Feature 2

    Value 1: Action would most likely take the agent from
             state s into a location where the agent could die.
    Value 0: Otherwise.
    """
    arena = state['arena']
    bombs = state['bombs']
    others = state['others']
    explosions = state['explosions']
    x, y, _, _ = state['self']

    # TODO: We can replace this stuff later with a directions array,
    # and have feature_2 return a vector for each action.
    if action == 'UP':
        x_next, y_next = x, y-1
    elif action == 'DOWN':
        x_next, y_next = x, y+1
    elif action == 'LEFT':
        x_next, y_next = x-1, y
    elif action == 'RIGHT':
        x_next, y_next = x+1, y
    elif action == 'BOMB' or action == 'WAIT':
        x_next, y_next = x, y

    # Check if the tile reached by the next action is occupied by an
    # object. This object may be destroyed by bombs, but prevents us
    # from moving into a free tile.
    if not tile_is_free(x_next, y_next, arena, bombs, active_agents):
        # Set the next action as equivalent to waiting. (Such an
        # action is invalid, as handled by a different feature.)
        x_next, y_next = x, y

    # Check if the agent will move into the blast range of an existing
    # bomb. This information is only available when the bomb has
    # already exploded, and has to be calculated manually when not.
    if explosions[(x_next, y_next)] > 0:
        return 1

    # Check if the agent will move into the blast range of a bomb that
    # explodes on the next time step. As the 'explosions' array does
    # not yet contain the blast range, we compute it manually by
    # iterating over each bomb.
    for xb, yb, t in bombs:
        if timer < 0 and (x_next, y_next) in get_blast_coords(xb, yb, arena):
            return 1

    # If the above conditions are not satisfied, the agent is most
    # likely safe from death by taking the given action.
    return 0


def feature_3(state, action):
    """Feature 3

    Value 1: Action most likely takes the agent from state s into a
             location where there is an obstacle (wall, crate, bomb or
             other agent in the arena).
    Value 0: Otherwise.
    """
    # Retrieve agent position.
    arena = state['arena']
    bombs = state['bombs']
    others = state['others']
    x, y, _, _ = state['self']

    if action == 'INVALID_ACTION':
        # Taking the agent into a wall is one possible invalid action.
        return 1
    elif action == 'UP':
        if not tile_is_free(x, y-1, arena, bombs, others):
            return 1
    elif action == 'DOWN':
        if not tile_is_free(x, y+1, arena, bombs, others):
            return 1
    elif action == 'LEFT':
        if not tile_is_free(x-1, y, arena, bombs, others):
            return 1
    elif action == 'RIGHT':
        if not tile_is_free(x+1, y, arena, bombs, others):
            return 1
    # No obstacle found, or the next action is not a movement action.
    return 0


def feature_4(state, action):
    """Feature 4

    Reward the minimal distance to move the agent away from a zone (of
    certain radius) containing a bomb.
    """
    pass


def feature_5(state, action):
    """Feature 5

    Value 1: Action is most likely to destroy a crate.
    Value 0: Otherwise.
    """
    # Retrieve arena and agent information.
    arena = state['arena']
    x, y, _, bombs_left = state['self']

    # Only the BOMB action allows the agent to destroy a coin.
    if action == 'BOMB' and bombs_left > 0:
        # Check if a nearby crate is inside the blast radius
        # of a dropped bomb. An agent drops a bomb at its
        # current location.
        return 1


def feature_6(state, action):
    """Feature 6

    Value 1: Action most likely takes the agent from state s
             into a location where there is a coin.
    Value 0: Otherwise.
    """

    # Retrieve agent and coin information.
    coins = state['coins']
    x, y, _, _ = state['self']

    # Only move actions (UP, DOWN, LEFT, RIGHT) allow the agent
    # to collect a coin.
    for coin in coins:
        if action == 'UP':
            if x == coin.x and (y)+1 == coin.y:
                return 1
        elif action == 'DOWN':
            if x == coin.x and (y)-1 == coin.y:
                return 1
        elif action == 'LEFT':
            if y == coin.y and (x)-1 == coin.x:
                return 1
        elif action == 'RIGHT':
            if y == coin.y and (x)+1 == coin.x:
                return 1

    # No coin was collected.
    return 0


def feature_7(state, action):
    """Feature 7

    Value: The distance between our agent and the nearest
           reachable crate.
    """
    pass


def feature_8(state, action):
    """Feature 8

    Value 1: Action is most likely to kill another agent.
    Value 0: Otherwise.

    Compared to crates, the difficulty here is that agents can
    move and drop bombs.
    """
    pass


def feature_8(state, action):
    """Feature 8, 9, 10

    Value: The distance between our agent's position and
           one of the other agents.
    """
    pass
