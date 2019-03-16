def TDLambdaLinFApp(Lambda, state, reward, state_next, w, z, f, step_size=0.05, discount=1):
    """TD(Lambda) with function approximation.

    This function implements the TD(Lambda) algorithm with linear function
    approximation. It must be called after each transition.

    Parameters:
    * state:      Array representing the last state.
    * state_next: Array representing the next state.
    * reward:     real number associated to the state transition.
    * w:          Array (d-dimensional) representing the parameter vector
                  of the linear function approximation.
    * z:          Array storing the (approximated) eligibility traces.
    * f:          Callable representing a feature extraction method. It should have the
                  same dimension as the parameter vector.
    * step_size:  Small non-negative real numbers chosen by the user.
    * discount:   The discount factor in the MDP.

    Return Value:
    * (w', z'):   Update of the parameter vector and eligibility traces.
    """
    f_y = f(state_next)
    f_x = f(state)
    delta = reward + discount*np.dot(w, f_y) - np.dot(w, f_x)
    
    # update elegibility traces and parameter
    z = f_x + discount*Lambda*z
    w = w + step_size*delta*z
    
    return (w, z)


def QLearningLinFApp(state, action, reward, state_next, w, f, step_size=0.05, discount=1):
    """Q-learning with linear function approximation

    This function implements the Q-learning algorithm with linear function
    approximation.  It must be called after each state transition.

    Parameters:
    * state:       Array representing the last state s.
    * action:      Action A which occured in the state transtion from s to s'.
    * reward:      Real number r associated to the state transition.
    * state_next:  Array representing the next state s'.
    * w:           Array (d-dimensional) representing the parameter vector of the
                   linear function approximation.
    * f:           Callable representing the feature extraction function.
    * step_size:   Small non-negative real numbers chosen by the user. For an optimal
                   choice of step size, see [Szepesvári 2009, p.20). 
                   (Example: c/sqrt(t), with the current step and c>0)
    * discount:    The discount factor in the MDP. In an episodic MDP (an MDP with
                   terminal states), we often consider discount=1.
  
    Return Value:
    * w':          Update of the parameter vector w using gradient descent.
    """
    # Compute the feature approximation for the last state s and action a.
    features = f(state, action)
    
    # Compute the maximum Q value for each action in the action space
    # (global parameter).
    Q_max, A_max = 0, None

    for A in actions:
        Q = np.dot(w, f(state_next, A))
        if Q > Q_max:
            Q_max, A_max = Q, A

    # Update intermediary values
    delta = reward + (discount * Q_max) - np.dot(w, features)

    # Return updated parameter vector
    return w + (step_size * delta * features)


def SARSALambdaLinFApp(Lambda, state, action, reward, state_next, action_next, w, z, f,
                       step_size=0.05, discount=1):

    """SARSA(Lambda) algorithm with linear function approximation

    This function implements the Sarsa(Lambda) algorithm with linear function
    approximation. It must be called after each transition.

    Parameters:
    * Lambda:      The hyperparameter for the algorithm.
    * state:       Array representing the last state s.
    * action:      Action A which occured in the state transtion from s to s'.
    * reward:      Real number r associated to the state transition.
    * state_next:  Array representing the next state s'.
    * action_next: Action A' chosen for the next state transition from s'.
    * w:           Array (d-dimensional) representing the parameter vector of the
                   linear function approximation.
    * z:           Array (d-dimensional) representing the vector of eligibility traces.

    Return Value:
    * (w', z'):    Update of the parameter vector and eligiblity traces.
    """
    features = f(state, action)
    features_next = f(state_next, action_next)

    # Compute eligibility traces.
    delta = reward + discount * np.dot(w, features_next) - np.dot(w, features)
    z = features + discount * Lambda * z

    return w + (step_size * delta * z), z
