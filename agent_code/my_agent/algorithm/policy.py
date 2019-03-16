# TODO: Check description (Koethe lecture etc.)
def QLearningGreedyPolicy(state, w, f):
    """Greedy policy for Q learning

    This function implements a greedy policy based on the highest
    Q-value. This policy can be used as one of the possible policies
    in to update the parameter vector (exploration), or as an optimal
    policy if the parameter vector is trained.

    Input Parameters:
    * state:  Array representing the last state s.
    * w:      Array (d-dimensional) representing the parameter vector of the
              linear function approximation.
    * f:      Callable representing the feature extraction function.
    """
