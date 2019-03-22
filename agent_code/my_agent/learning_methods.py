import numpy as np


"""
Learning schedules (compute time steps for iterative methods)
"""
def learning_schedule_1(time_step, c=1.0, eta=0.5):
    return c / pow(time_step, eta)

def learning_schedule_2(time_step):
    return 10.0 / (9.0 + time_step)

def learning_schedule_3(time_step):
    return 0.1

def learning_schedule_4(time_step, max_steps=400):
    if (1 <= time_step < max_steps/4):
        return 0.1
    elif (max_steps/4 <= time_step < max_steps/2):
        return 0.05
    elif (max_steps/2 <= time_step <= max_steps/4):
        return 0.01


"""
Learning methods (Q-Learning, SARSA)
"""
def UpdateWeightsLFA(X, A, R, Y, weights, alpha=0.1, discount=0.95):
    """Update the weight vector w for Q-learning with semi-gradient descent.
    
    The features X and Y are assumed to have state_action(action) and
    max_q(weights) methods available. See feature_extraction.py for details.
    """
    X_A = X.state_action[A]
    Q_max, A_max = Y.max_q(weights)
    TD_error = R + (discount * Q_max) - np.dot(weights, X_A)

    return weights + (alpha * TD_error * X_A)


def UpdateWeightsLFA_DoubleQ(X, A, R, Y, weights1, weights2, alpha=0.1, discount=0.95):
    """
    Update the weight vectors w1, w2 for Double Q-learning.
    """
    Q_choice = np.random_choice(['tails', 'heads'])
    X_A = X.state_action[A]

    if Q_choice == 'heads':
        # update weights1 on heads, use weights2 for approximation
        Q2 = np.dot(weights2, Y.max_q(weights1))
        TD_error = R + (discount * Q2) - np.dot(weights1, X_A)

        return weights1 + (alpha * TD_error * X_A), weights2
    else:
        # update weights2 on tails, use weights1 for approximation
        Q1 = np.dot(weights1, Y.max_q(weights2))
        TD_error = R + (discount * Q1) - np.dot(weights2, X_A)

        return weights1, weights2 + (alpha * TD_error * X_A)


def UpdateWeightsLFA_SARSA(X, A, R, Y, B, weights, traces, Lambda=1,
                           alpha=0.1, discount=0.95):
    """Update the weight vector w using SARSA.

    The features X and Y are assumed to have state_action(action) and
    max_q(weights) methods available. See feature_extraction.py for details.
    """
    X_A = X.state_action[A]
    Y_B = Y.state_action[B]    

    TD_error = R + discount * np.dot(weights, Y_B) - np.dot(weights, X_A)
    traces_n = X_A + discount * Lambda * eligibility_traces

    return weights + alpha * TD_error * traces_n, traces_n
