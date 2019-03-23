def UpdateWeightsLFA_DoubleQ(X, A, R, Y, weights1, weights2, alpha=0.1, discount=0.95):
    """
    Update the weight vectors w1, w2 for Double Q-learning.
    """
    Q_choice = np.random_choice(['tails', 'heads'])
    X_A = X.state_action(A)

    if Q_choice == 'heads':
        # update weights1 on heads, use weights2 for approximation
        # TODO: max_q returns Q_max (real) and A_max (vector)
        Q2 = np.dot(weights2, Y.max_q(weights1))
        TD_error = R + (discount * Q2) - np.dot(X_A, weights1)

        return weights1 + (alpha * TD_error * X_A), weights2
    else:
        # update weights2 on tails, use weights1 for approximation
        # TODO: max_q returns Q_max (real) and A_max (vector)
        Q1 = np.dot(weights1, Y.max_q(weights2))
        TD_error = R + (discount * Q1) - np.dot(X_A, weights2)

        return weights1, weights2 + (alpha * TD_error * X_A)


def UpdateWeightsLFA_SARSA(X, A, R, Y, B, weights, z, Lambda=0.5, alpha=0.1, discount=0.95):
    """Update the weight vector w using SARSA.

    The features X and Y are assumed to have state_action(action) and
    max_q(weights) methods available. See feature_extraction.py for details.
    """
    X_A = X.state_action(A)
    Y_B = Y.state_action(B)

    TD_error = R + discount * np.dot(Y_B, weights) - np.dot(X_A, weights)
    z_next = X_A + discount * Lambda * z

    return weights + alpha * TD_error * z_next, z_next


# TODO: The performance of this feature appears lacking in spite of obvious
# nature... maybe due to look_for_targets_path function?
def feature12(self):
    """
    Penalize the agent for moving directly into a dead end, if a bomb was placed
    as the previous action.
    """
    path_dead_end = look_for_targets_path(self.free_space, self.agent, self.dead_ends)
    feature = []

    # Check if there are dead ends in the arena.
    if len(path_dead_end) == 0:
        return [0] * len(self.actions)

    for action in self.actions:
        if action == 'BOMB' or action == 'WAIT':
            # Note that placing a bomb if one has been placed previously (the
            # scenario envisioned by this feature), that the agent performs an
            # invalid agent. However, this is already covered by a different
            # feature.
            feature.append(0)
        elif len(path_dead_end) == 1:
            d = self.directions[action]

            # If a bomb was placed in the previous step, it is in the same tile
            # the agent is located in.
            if d == path_dead_end[0] and self.agent in self.bombs_xy:
                feature.append(1)
            else:
                feature.append(0)
        else:
            # Do not penalize a longer path towards a dead end.
            feature.append(0)

    return feature


# TODO: This is a somewhat risky move which may cause the agent to blow itself up...
def feature13(self, coins_limit, crates_limit, radius=3):
    """Hunting mode

    Place a bomb depending on distance and amount of agents in the vicinity of
    the agent.
    """
    feature = []
    target_agents = 0

    if len(self.coins) > coins_limit or len(self.crates) > crates_limit:
        return [0] * len(self.actions)

    for other in self.others_xy:
        # The amount of steps the agent has to take for reaching the current
        # opposing agent.
        path_to_other = look_for_targets_path(self.free_space, self.agent, self.others_xy)
        print("PATH TO AGENT", path_to_other, sep=" ")

        if len(path_to_other) <= radius:
            target_agents += 1

    for action in self.actions:
        if action == 'BOMB':
            feature.append(target_agents)
        else:
            feature.append(0)

    return feature
                    
