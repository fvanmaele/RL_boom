import numpy as np
import os # required for training
from random import shuffle

from agent_code.my_agent.feature_extraction import *
from agent_code.my_agent.learning_methods import *
from settings import e
from settings import s

## TRAINING PARAMETERS

# Allow to check various combinations of training methods in parallel
# through multiple agent processes by taking values from the
# environment.

# TODO: only enable in training mode (move to reward_update?)
t_learning_schedule = os.environ.get('MRBOMBASTIC_ALPHA')
t_policy = os.environ.get('MRBOMBASTIC_POLICY')
t_policy_eps = os.environ.get('MRBOMBASTIC_POLICY_EPS')
t_weight_begin = os.environ.get('MRBOMBASTIC_WEIGHTS_BEGIN')

# Default values for training.
if t_learning_schedule == None: t_learning_schedule = 'fixed'
if t_policy == None: t_policy = 'greedy'
if t_policy_eps == None: t_policy_eps = 0.1
if t_weight_begin == None: t_weight_begin = 'bestguess'

# Construct unique id for naming of persistent data.
# TODO: include eps?
t_training_id = "{}_{}_{}".format(t_learning_schedule, t_policy, t_weight_begin)

# Save weights to file
print("TRAINING ID:", t_training_id)
weights_file = "weights_{}.npy".format(t_training_id)


## REWARDS AND POLICY

def initialize_weights(method, dim):
    if method == 'bestguess':
        return np.asarray([1, 1.5, -7, -1, 4, -0.5, 1.5, 1, 0.5, 0.5, 0.8, 0.5, 1, -1])
    elif method == 'ones':
        return np.ones(dim)
    elif method == 'zero':
        return np.zeros(dim)
    elif method == 'random':
        return np.random.rand(dim, 1)
    else:
        return None


# TODO: use diminishing eps-greedy policy in learning
def policy_select_action(greedy_action, policy, policy_eps, game_ratio, logger=None):
    # Select a random action for use in epsilon-greedy or random-walk
    # exploration.
    random_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])

    if policy == 'greedy':
        logger.info('Pick greedy action %s', greedy_action)
        return greedy_action
    elif policy == 'epsgreedy':
        logger.info('Picking greedy action at probability %s', 1-policy_eps)
        return np.random.choice([greedy_action, random_action], p=[1-policy_eps, policy_eps])
    elif policy == 'diminishing':
        eps_dim = min(0.05, policy_eps * (1 - game_ratio))
        logger.info('Picking greedy action at probability %s', eps_dim)
        return np.random.choice([greedy_action, random_action], p=eps_dim)
    else:
        return None

    
def new_alpha(method, time_step, c=0.1, eta=1, max_steps=400):
    if method == 'quotient1':
        return c / pow(time_step, eta)
    elif method == 'quotient2':
        return 10.0 / (9.0 + time_step)
    elif method == 'fixed':
        return c
    elif method == 'subdivision':
        if (1 <= time_step < max_steps/4):
            return 0.1
        elif (max_steps/4 <= time_step < max_steps/2):
            return 0.05
        elif (max_steps/2 <= time_step <= max_steps/4):
            return 0.01
    else:
        return None


def new_reward(events):
    # An action is always punished with a small negative reward due to
    # time constraints on the game.
    reward = -1

    for event in events:
        if event == e.BOMB_DROPPED:
            reward += 1
        elif event == e.COIN_COLLECTED:
            reward += 200 # 100?
        elif event == e.KILLED_SELF:
            reward -= 300
        elif event == e.CRATE_DESTROYED:
            reward += 10
        elif event == e.COIN_FOUND:
            reward += 50
        elif event == e.KILLED_OPPONENT:
            reward += 300
        elif event == e.GOT_KILLED:
            reward -= 300
        elif event == e.SURVIVED_ROUND:
            reward += 100            
        elif event == e.INVALID_ACTION:
            reward -= 2
        # elif event == e.WAITED:
        #     reward += 10

    return reward


## LEARNING METHODS

"""
Learning methods (Q-Learning, SARSA)
"""
def UpdateWeightsLFA(X, A, R, Y, weights, alpha=0.1, discount=0.95):
    """Update the weight vector w for Q-learning with semi-gradient descent.
    
    The features X and Y are assumed to have state_action(action) and
    max_q(weights) methods available. See feature_extraction.py for details.
    """
    X_A = X.state_action(A)
    Q_max, _ = Y.max_q(weights)
    TD_error = R + (discount * Q_max) - np.dot(X_A, weights)

    return weights + (alpha * TD_error * X_A)


## GAME METHODS

def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """    
    np.random.seed()    

    # Reward for full game (10 rounds)
    self.current_round = 1
    self.accumulated_reward = 0

    # Tuples (S,A,R,S') for later sampling.
    self.replay_buffer = []
    self.discount = 0.8

    # Values for first game.
    self.current_game = 1

    try:
        self.weights = np.load(weights_file)
        print("LOADED WEIGHTS", self.weights)
    except EnvironmentError:
        print("INITIALIZING WEIGHTS")
        self.weights = initialize_weights(t_weight_begin, 14)        
    print(self.weights, sep=" ")

    # Load persistent data for subsequent games, depending on training ID.

    # Keep copy of loaded weights for optimization problem (see lecture).
    #self.weights_episode = self.weights


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
    # We need to perform at least one action before computing an
    # intermediary reward.
    if self.game_state['step'] == 1:
        return None

    reward = new_reward(self.events)
    self.logger.info('Given reward of %s', reward)
    self.accumulated_reward += reward

    # Extract features from the new ("next") game state.
    self.F = RLFeatureExtraction(self.game_state)
    print("PREVIOUS STATE:", self.F_prev.state().T)
    print("NEXT STATE:", self.F.state().T)

    # Get action of the previous step. The previous action was set in
    # the last act() call and is thus named 'next_action'.
    self.prev_action = self.next_action

    # Keep track of all experiences in the episode for later learning.
    self.replay_buffer += (self.F_prev, self.prev_action, reward, self.F)


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
    # Update features to for the current ("previous") game state.
    self.F_prev = RLFeatureExtraction(self.game_state)

    # Multiple actions may give the same (optimal) reward. To avoid bias towards
    # a particular action, shuffle the greedy actions.
    Q_max, A_max = self.F_prev.max_q(self.weights)        
    shuffle(A_max)

    # TODO: Move to reward_update (to allow SARSA use), only use exploitation
    # strategy (greedy) here for known weights
    self.next_action = policy_select_action(A_max[0], t_policy, t_policy_eps, 1.0, self.logger)


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    # TODO: Write accumulated reward to files depending on training_id

    print("{} {}".format(self.current_round, self.accumulated_reward))
    self.accumulated_reward = 0

    # TODO: update weights at end of full game (10 rounds) to reduce bias (temporary w')
    #np.save(weights_file, self.weights)
    # Default hyperparameters: c=0.1, eta=0.1, max_steps=400
    self.alpha = new_alpha(t_learning_schedule, self.game_state['step']-1)
    print("ALPHA:", self.alpha)

    # # Set learning algorithm (Q/SARSA) depending on learning parameters.
    #self.weights = UpdateWeightsLFA(self.F_prev, self.prev_action, reward, F,
    #                                self.weights, self.alpha, self.discount)

    #self.current_round += 1

