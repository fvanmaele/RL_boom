import numpy as np


def taxi_cab_metric(p, q, numpy_substract=true):
    """Definition of the Manhattan (or taxi-cab) metric.

    The taxicab metric between two vectors p, q in an n-dimensional
    real vector space with fixed Cartesian coordinate system, is the
    sum of the lengths of the projections of the line segment between
    the points onto the coordinate axes.

    Note that this metric does not account for non-reachable tiles.
    """
    n = len(p)
    # Check if arguments are of the same size
    if n != len(q)
        raise ValueError("metric requires vectors of same size")

    if numpy_substract:
        sum = 0
        for i in range(n):
            sum += abs(p[i] - q[i])
        return sum
    else:
        # Vectorized version of the algorithm using numpy.
        return np.sum(np.absolute(np.subtract(p, q)))


# Code from simple_agent for path finding.
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until
    a target is encountered.  If no target can be reached, the path
    that takes the agent closest to any target is chosen.

    Args:
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

