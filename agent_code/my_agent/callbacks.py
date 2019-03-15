import numpy as np


# TODO: This metric does not account for non-reachable tiles.
# Adapt 'look_for_targets' to return the "distance" as the amount
# of steps required to collect a coin.
def taxi_cab_metric(p, q, numpy_substract=true):
    """Definition of the Manhattan (or taxi-cab) metric.

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


class FeatureExtraction:
    """
    Generic class for feature extraction based on a state s.

    Variables:
    * actions: Possible actions in the Bomberman game.
    * d:       Length of the array representing the feature vector.
    """
    def __init__(self, d):
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
        self.dim = d

    """
    Return a feature vector from a given state. Implementation should
    follow in derived classes.
    """
    def __call__(self, state, action):
        raise AttributeError("Feature extraction undefined")


class FEBinary(FeatureExtraction):
    """
    Derived class implementing feature extraction using 3 binary
    features and 3 scalar features.
    """
    def __init__(self):
        FeatureExtraction.__init__(self, 10)
        self.result = [None] * 10

    # TODO: use return values for feature_i instead of populating
    # class variable
    def feature_1(self, state, action):
        """Feature 1

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
        # Begin at unthreatened state.
        result[0] = 0

        # Retrieve arena, agent and bomb information
        self.arena = state['arena']
        self.bombs = state['bombs']
        self.x, self.y, _, _ = state['self']

        # All placed bombs (by several agents, ourselves included) are
        # considered in succession.
        for xb, yb, timer in self.bombs:
            if timer < 0:
                # Bomb will explode on the next action, unless the agent
                # performs an evasive maneuver, or is not in the blast
                # range of the bomb.
                blast_range_h = []
                blast_range_v = []

                # Check three tiles up, left, right and down of bomb
                # for a wall. Items (crates, other agents) possibly
                # destroyed by the bomb are not taken into account.
                for i in range(1, 5): # {1,..,4}
                    if self.arena[xb-i, yb] == -1:
                        blast_range_h.append((xb-i+1, yb))
                        break
                for i in range(1, 5):
                    if self.arena[xb+i, yb] == -1:
                        blast_range_h.append((xb+i-1, yb))
                        break
                for i in range(1, 5):
                    if self.arena[xb, yb-i] == -1:
                        blast_range_v.append((xb, yb-i+1))
                        break
                for i in range(1, 5):
                    if self.arena[xb, yb+i] == -1:
                        blast_range_v.append((xb, yb+i-1))
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
                # action allows to perform an evasive maneuver.
                if (blast_range_h[0] <= self.x <= blast_range_h[-1] and
                    blast_range_v[0] <= self.y <= blast_range_v[-1]):
                    if (action == 'WAIT' or action == 'BOMB' or action == 'INVALID_ACTION'):
                        result[0] = 1
                    elif action == 'UP':
                        if blast_range_v[0] <= (self.y)+1 <= blast_range_v[-1]:
                            result[0] = 1
                    elif action == 'DOWN':
                        if blast_range_v[0] <= (self.y)-1 <= blast_range_v[-1]:
                            result[0] = 1
                    elif action == 'LEFT':
                        if blast_range_h[0] <= (self.x)-1 <= blast_range_h[-1]:
                            result[0] = 1
                    elif action == 'RIGHT':
                        if blast_range_h[0] <= (self.x)+1 <= blast_range_h[-1]:
                            result[0] = 1
                    else:
                        raise AttributeError("invalid action for bomb maneuver")
            else:                
                continue # Bomb will explode in some later time step.


        def feature_2(self, state, action):
            """Feature 2

            Value 1: Action most likely takes the agent from
                     state s into a location where there is a wall.
            Value 0: Otherwise.
            """
            # Begin at free tile.
            result[1] = 0

            # Retrieve agent position.
            self.x, self.y, _, _ = state['self']

            # Taking the agent into a wall is one possible invalid action.
            if action == 'INVALID_ACTION':
                result[1] = 1
            elif action == 'WAIT' or action == 'BOMB':
                # Placing a bomb or waiting does not result in movement.
                result[1] = 0
            elif action == 'UP':
                if self.arena[self.x, (self.y)+1] == -1:
                    result[1] = 1
            elif action == 'DOWN':
                if self.arena[self.x, (self.y)-1] == -1:
                    result[1] = 1
            elif action == 'LEFT':
                if self.arena[self.x, (self.x)-1] == -1:
                    result[1] = 1
            elif action == 'RIGHT':
                if self.arena[self.x, (self.x)-1] == -1:
                    result[1] = 1


        def feature_3(self, state, action):
            """Feature 3

            Value 1: Action most likely takes the agent from
                     state s into a location where there is a coin.
            Value 0: Otherwise.
            """
            # Begin with no coin.
            result[2] = 0

            # Retrieve agent and coin information.
            self.coins = state['coins']
            self.x, self.y, _, _ = state['self']

            # Only move actions (UP, DOWN, LEFT, RIGHT) allow the agent
            # to collect a coin.
            for coin in self.coins:
                if action == 'UP':
                    if self.x == coin.x and (self.y)+1 == coin.y:
                        result[2] = 1
                elif action == 'DOWN':
                    if self.x == coin.x and (self.y)-1 == coin.y:
                        result[2] = 1
                elif action == 'LEFT':
                    if self.y == coin.y and (self.x)-1 == coin.x:
                        result[2] = 1
                elif action == 'RIGHT':
                    if self.y == coin.y and (self.x)+1 == coin.x:
                        result[2] = 1
                else:
                    pass

                
        def feature_4(self, state, action):
            """Feature 4

            Value: The distance between our agent and the nearest 
                   reachable coin.
            """
            # Begin with negative value (no agent).
            result[3] = -1
            # TODO: get list of available crates (cf. simple_agent/callbacks.py)
            # TODO: compute blast range as in feature 1


        def feature_5(self, state, action):
            """Feature 5

            Value 1: Action is most likely to destroy a crate.
            Value 0: Otherwise.
            """
            # Begin with untouched crate.
            result[4] = 0

            # Retrieve arena and agent information.
            self.arena = state['arena']
            self.x, self.y, _, self.bombs_left = state['self']

            # Only the BOMB action allows the agent to destroy a coin.
            if action == 'BOMB' and self.bombs_left > 0:
                # Check if a nearby crate is inside the blast radius
                # of a dropped bomb. An agent drops a bomb at its
                # current location.
                
                result[4] = 1


        def feature_6(self, state, action):
            """Feature 6

            Value: The distance between our agent and the nearest
                   reachable crate.
            """


        def feature_7(self, state, action):
            """Feature 7

            Value 1: Action is most likely to kill another agent.
            Value 0: Otherwise.

            Compared to crates, the difficulty here is that agents can
            move and drop bombs.
            """


        def feature_8(self, state, action):
            """Feature 8, 9, 10

            Value: The distance between our agent's position and
                   one of the other agents.
            """


class QLearning:
    """
    Class for Q-learning methods
    """
    pass:


def QLearningLinFApp(state, action, reward, state_next, w, f, step_size=0.05, discount=1):
    """Q-learning with linear function approximation

    This function implements the Q-learning algorithm with linear function
    approximation.  It must be called after each state transition.

    Input Parameters:
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
    Q_max = 0
    A_max = None

    for A in actions:
        Q = np.dot(w, f(state_next, A))

        if Q > Q_max:
            Q_max, A_max = Q, A

    # Update intermediary values
    delta = reward + (discount * Q_max) - np.dot(w, features)

    # Return updated parameter vector
    return w + (step_size * delta * features)


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
    

def SARSALambdaLinFApp(Lambda, state, action, reward, state_next, action_next, w, z, f,
                       step_size=0.05, discount=1):

    """SARSA(Lambda) algorithm with linear function approximation

    This function implements the Sarsa(Lambda) algorithm with linear function
    approximation. It must be called after each transition.

    Input Parameters:
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
    * (w', z'):    Update of w and z.
    """
    features = f(state, action)
    features_next = f(state_next, action_next)

    # Compute eligibility traces.
    delta = reward + discount * np.dot(w, features_next) - np.dot(w, features)
    z = features + discount * Lambda * z

    return w + (step_size * delta * z), z


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    np.random.seed()
    self.reward_sequence = []

    # We require the previous state to implement the Q-learning algorithm
    # with linear function approximation.
    self.prev_state = self.game_state


def act(self):
    """Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via self.game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    Set the action you wish to perform by assigning the relevant string to
    self.next_action. You can assign to this variable multiple times during
    your computations. If this method takes longer than the time limit specified
    in settings.py, execution is interrupted by the game and the current value
    of self.next_action will be used. The default value is 'WAIT'.
    """
    np.random_seed()

    # Load parameter vector w for Q-learning with function approximation.
    if os.path.isfile('w.pickle'):
        with open('w.pickle', 'rb') as file:
            self.w = pickle.load(file)
    else:
        # Initialize w with values of 1. The size of the parameter
        # vector depends on the amount of features in the feature
        # extraction function f.
        self.w = np.fill(feature_extraction.dim, 1)


def reward_update(self):
    """Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occured during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state. In
    contrast to act, this method has no time limit.

    @fvanmaele: We transform action consequences into something that Q-learning
    can use to learn the Q-function by giving in-game events a numerical reward.
    For learning the optimal behavior, the rewards of different objectives should
    be set carefully so that maximizing the obtained rewards results in the
    desired behavior.

    The following rewards have been chosen to clearly distinct between good and
    bad actions. Dying is represented by a very negative reward. The reward of
    killing a player is attributed to the player that actually placed the involved
    bomb. The rest of the rewards promote active behavior. No reward is given to
    finally winning the game (when all other players died).

    In order to maximize the total reward intake, the agent should learn not to
    die, and kill as many opponents and break most crates with its bombs.
    """
    # Performing an action is always punished with a small negative reward.
    reward = -1

    for event in agent.game_state['events']:
        if event == e.BOMB_DROPPED:
            # We give no special reward to the action of dropping a
            # bomb, to leave flexibility in the chosen strategies.
            reward += 0
        elif event == e.COIN_COLLECTED:
            # Collecting coins is the secondary goal of the game.
            reward += 100
        elif event == e.KILLED_SELF:
            # Killing ourselves through bomb placement is something we
            # wish to avoid.
            reward -= 100
        elif event == e.KILLED_OPPONENT:
            # Killing opponents is the primary goal of the game.
            reward += 300
        elif event == e.GOT_KILLED:
            # Dying at the hands of an opponent is classified worse as
            # death by self.
            reward -= 300
        elif event == e.WAITED or event == e.INVALID_ACTION:
            # An invalid action (such as bumping into a wall) is
            # equivalent to performing no action at all. Punish both
            # with a small negative reward.
            reward -= 2

    # We keep track of all intermediary rewards in the episode
    agent.logger.info(f'Given reward of {reward}')
    agent.reward_sequence.append(reward)

    # Update the weights (feature vector) after each intermediary step
    # in the game world. This requires the "previous" action, oddly
    # named next_action in the bomberman framework.
    prev_action = self.next_action


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    pass
