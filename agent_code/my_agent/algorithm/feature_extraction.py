# Save all actions in the action space in a global variable for
# simplicity.
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

def bulk_feature(feat, state):
    results = []

    for a in actions:
        results.append(feat(state, action))
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

    The only way for an agent to die is by crossing an exploding
    bomb.  The rules are the following:

    "Once a bomb is dropped, it will detonate after four steps and
    crate an explosion that extends three tiles up, down, left and
    right.  The explosion destroys crates and agents, but will
    stop at stone walls and does not reach around corners. Agents
    can only drop a new bomb after their previous one has
    exploded."

    Performing an "evasive maneuver" (that is, taking the agent
    from a threatened state s into a location where the agent is
    no longer likely to die) is a special case of assigning 0.
    """
    # Retrieve arena, agent and bomb information
    arena = state['arena']
    bombs = state['bombs']
    x, y, _, _ = state['self']

    # All placed bombs (by several agents, ourselves included) are
    # considered in succession.
    for xb, yb, timer in bombs:
        if timer < 0:
            # Bomb will explode on the next action, unless the agent
            # performs an evasive maneuver, or is not in the blast
            # range of the bomb.
            blast_range_h = []
            blast_range_v = []

            # Check three tiles up, left, right and down of bomb
            # for a wall. Crates and other agents are destroyed by
            # the bomb and do not affect the blast range.
            # 
            # TODO: get_bomb_coords in item.py (bomb class) is
            # very similar, apart from using a single list
            # instead of two.
            for i in range(1, 5): # {1,..,4}
                if arena[xb-i, yb] == -1: # left
                    blast_range_h.append((xb-i+1, yb))
                    break
            for i in range(1, 5):
                if arena[xb+i, yb] == -1: # right
                    blast_range_h.append((xb+i-1, yb))
                    break
            for i in range(1, 5):
                if arena[xb, yb+i] == -1: # down (reversed)
                    blast_range_v.append((xb, yb+i-1))
                    break
            for i in range(1, 5):
                if arena[xb, yb-i] == -1: # up (reversed)
                    blast_range_v.append((xb, yb-i+1))
                    break

            # TEST: A bomb cannot be fully enclosed by walls and
            # must have a blast range.
            if len(blast_range_h) == len(blast_range_v) == 1:
                raise ValueError("bomb may not be enclosed by walls")
            if len(blast_range_h) == 0 or len(blast_range_v) == 0:
                raise ValueError("bomb without blast range")

            # TEST: Check if range of blast radius is correct.
            if not (1 <= blast_range_h[-1] - blast_range_h[0] <= 7 and
                    1 <= blast_range_v[-1] - blast_range_v[0] <= 7):
                raise ValueError("bomb has invalid blast range")

            # If the agent is in the blast range, check if the
            # action allows to perform an evasive maneuver. While
            # crates do not provide cover from the blast, they
            # block a free tile where the agent may otherwise move
            # for cover.
            if (blast_range_h[0] <= x <= blast_range_h[-1] and
                blast_range_v[0] <= y <= blast_range_v[-1]):
                if (action == 'WAIT' or action == 'BOMB' or action == 'INVALID_ACTION'):
                    return 1 # Agent threat found
                elif (action == 'UP' and not arena[x, y-1] == 1):
                    if blast_range_v[0] <= (y)-1 <= blast_range_v[-1]:
                        return 1
                elif (action == 'DOWN' and not arena[x, y+1] == 1):
                    if blast_range_v[0] <= (y)+1 <= blast_range_v[-1]:
                        return 1
                elif (action == 'LEFT' and not arena[x-1, y] == 1):
                    if blast_range_h[0] <= (x)-1 <= blast_range_h[-1]:
                        return 1
                elif (action == 'RIGHT' and not arena[x+1, y] == 1):
                    if blast_range_h[0] <= (x)+1 <= blast_range_h[-1]:
                        return 1
                else:
                    raise AttributeError("invalid action for bomb maneuver")
        else:                
            continue # Bomb will explode in some later time step.

        # No threats to the agent were found.
        return 0


def feature_3(state, action):
    """Feature 3

    Value 1: Action most likely takes the agent from state s into a
             location where there is an obstacle (wall or crate).
    Value 0: Otherwise.
    """
    # Retrieve agent position.
    arena = state['arena']
    x, y, _, _ = state['self']

    # Taking the agent into a wall is one possible invalid action.
    if action == 'INVALID_ACTION':
        return 1
    elif action == 'WAIT' or action == 'BOMB':
        # Placing a bomb or waiting does not result in movement.
        return 0
    elif action == 'UP':
        if arena[x, y-1] in (-1, 1):
            return 1
    elif action == 'DOWN':
        if arena[x, y+1] in (-1, 1):
            return 1
    elif action == 'LEFT':
        if arena[x-1, y] in (-1, 1):
            return 1
    elif action == 'RIGHT':
        if arena[x+1, y] in (-1, 1):
            return 1

    # No obstacle found.
    return 0


def feature_4(state, action):
    """Feature 4

    Reward the minimal distance to move the agent away from a zone (of
    certain radius) containing a bomb.
    """


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
