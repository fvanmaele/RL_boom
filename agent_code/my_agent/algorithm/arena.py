import numpy as np
import settings as game_settings


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
    power = game_settings.settings['bomb_power']
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
def look_for_targets(free_space, start, targets, logger=None):
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

    # Write debug output
    if logger:
        logger.debug(f'Suitable target found at {best}')

    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]

