import numpy as np

# TODO: This metric does not account for non-reachable tiles.
# Adapt 'look_for_targets' to return the "distance" as the amount
# of steps required to collect a coin.
def taxi_cab_metric(p, q, numpy_substract=true):
    """Definition of the Manhatten (or taxi-cab) metric.

    The taxicab metric between two vectors p, q in an n-dimensional
    real vector space with fixed Cartesian coordinate system, is the
    sum of the lengths of the projections of the line segment between
    the points onto the coordinate axes.
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


# TODO: Consider not only the distance between agent and coins,
# but the distance *between* separate coins as well?
def coin_feature_extraction(self):
    """Perform feature extraction for the coin-only scenario.

    The state of the game environment is described via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py
    and 'get_state' in agents.py to see what it contains.

    As the agent moves in a grid-like environment, use the manhattan
    (taxi-cab) metric to compute the distance between the agent and
    any available coins. The feature vector is then composed as a "radar":
    coins are put into balls of a certain radius.
    """
    arena = self.game_state['arena']
    coins = self.game_state['coins']
    x, y, _, bombs_left = self.game_state['self']

    # Compute distance between agent and each available coin.
    distance_vector = []
    for i in range(len(coins)):
        distance = taxi_cab_metric([x, y], [coins[i].x, coins[i].y])

    # TODO: Generate features with a given step size (e.g. N=1 for all
    # possible distances 1-28)
    feature_dist_05  = 0
    feature_dist_10  = 0
    feature_dist_15  = 0
    feature_dist_inf = 0

    for d in distance_vector:
        # By definition, any collectable coin has not already been collected.
        # TODO: When could the case "distance = 0" still occur?
        if d == 0:
            self.agent.logger.info(f'warning: empty distance of agent to coin')
        elif d <= 5:
            feature_dist_05 += d
        elif d <= 10:
            feature_dist_10 += d
        elif d <= 15:
            feature_dist_15 += d
        else:
            feature_dist_inf += d

    # Assemble feature vector
    return [feature_dist_05, feature_dist_10,
            feature_dist_15, feature_dist_inf]
