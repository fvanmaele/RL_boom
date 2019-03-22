import numpy as np


"""
Different approaches to compute the learning parameter alpha.
"""
def learning_schedule_1(time_step, c=1.0, eta=0.5):
    return c / pow(time_step, eta)

def learning_schedule_2(time_step):
    return 10.0 / (9.0 + time_step)

def learning_schedule_3(time_step):
    return 0.1

def learning_schedule_4(time_step, max_steps=400)
    if (1 <= time_step < max_steps/4):
        return 0.1
    elif (max_steps/4 <= time_step < max_steps/2):
        return 0.05
    elif (max_steps/2 <= time_step <= max_steps/4):
        return 0.01


def QMaxLFA(feature_matrix, weights):
    """Compute the maximum Q-value for all given actions using linear function
    approximation.

    In feature_matrix, each column represents a feature vector F_i(S,A), and each
    row represents an action. The maximum Q-value may be used to update weights
    during training, or to implement a greedy policy.
    """
    # Compute the dot product (w, F_i(S,A)) for every action.
    Q_lfa = np.dot(weights, feature_next)
    Q_max = np.max(Q_next)

    # Multiple actions may give the same (optimal) reward. To avoid bias towards a
    # particular action, shuffle the resulting array before return it.
    A_max = np.where(Q_lfa == Q_max)[0]
    A_max = shuffle(A_max)

    return Q_max, A_max


def UpdateWeightsLFA(state_action, reward, state_next, weights, alpha=0.1, discount=0.95):
    """Update the weight vector w using Q-learning with semi-gradient descent.

    state_action represents the feature vector F(S, A) with S the current
    state. state_next represents a matrix where every column is a feature F(S', A),
    with S' the next state of the agent.
    """
    Q_max, _ = QMaxLFA(state_next, weights)
    TD_error = reward + discount * Q_max - np.dot(weights, state_action)

    return w + alpha * TD_error * state_action


def UpdateWeightsLFA_SARSA(state_action, reward, state_action_next, weights, traces,
                           Lambda=1, alpha=0.1, discount=0.95):
    """Update the weight vector w using SARSA.

    state_action and state_action_next represent the feature vector F(S, A) and
    F(S', A), where S and S' are the current and next state, respectively.
    """
    TD_error = reward + discount * np.dot(weights, state_action_next) - np.dot(weights, state_action)
    traces_n = state_action + discount * Lambda * eligibility_traces

    return weights + alpha * TD_error * traces_n, traces_n


# TODO: Check description (Koethe lecture etc.)
def QLearningGreedyPolicy(state, actions, f, w):
    """Greedy policy for Q learning

    This function implements a greedy policy based on the highest
    Q-value. This policy can be used as one of the possible policies
    in to update the parameter vector (exploration), or as an optimal
    policy if the parameter vector is trained.
    """
    
