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
# TODO: only enable in training mode (move to reward_update)
t_batch_size = os.environ.get('MRBOMBASTIC_BATCH_SIZE')
t_learning_algorithm = os.environ.get('MRBOMBASTIC_LEARNING')
t_learning_schedule = os.environ.get('MRBOMBASTIC_ALPHA')
t_policy = os.environ.get('MRBOMBASTIC_POLICY')
t_policy_eps = os.environ.get('MRBOMBASTIC_POLICY_EPS')
t_weight_begin = os.environ.get('MRBOMBASTIC_WEIGHTS_BEGIN')

# Default values for training.
if t_batch_size == None: t_batch_size = 1
if t_learning_algorithm == None: t_learning_algorithm = 'qlearning'
if t_learning_schedule == None: t_learning_schedule = 'quotient1'
if t_policy == None: t_policy = 'epsgreedy'
if t_policy_eps == None: t_policy_eps = 0.2
if t_weight_begin == None: t_weight_begin = 'random'

# Construct unique id for naming of persistent data.
# TODO: include eps?
t_training_id = "N{}_{}_{}_{}_{}".format(t_batch_size, t_learning_algorithm,
                                         t_learning_schedule,
                                         t_policy, t_weight_begin)
print("TRAINING ID:", t_training_id)

# Save weights to file
weights_file = "weights_{}.npy".format(t_training_id)

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
    self.accumulated_reward = 0

    # Tuples (S,A,R,S') for later sampling.
    self.replay_buffer = []
    self.discount = 0.95

    # Values for first game.
    self.current_game = 1

    try:
        self.weights = np.load(weights_file)
        print("LOADED WEIGHTS", self.weights)
    except EnvironmentError:
        print("INITIALIZING WEIGHTS")
        if t_weight_begin == 'bestguess':
            self.weights = np.asarray([1.5, -4, -1, 3, -0.5, 1.5, 1, 1.5, 0.5, 1, 1.5, -3, 2])
        elif t_weight_begin == 'ones':
            self.weights = np.ones(13)
        elif t_weight_begin == 'zero':
            self.weights = np.zeros(13)
        elif t_weight_begin == 'random':
            self.weights = np.random.rand(13, 1)
            
    self.z = np.random.rand(13, 1)
    # Load persistent data for subsequent games, depending on training ID.

    # Keep copy of loaded weights for optimization problem (see lecture).
    #self.weights_episode = self.weights


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
    self.round = 0
    # Compute the action with highest Q-value.
    # TODO: Check for double Q-learning
    F = RLFeatureExtraction(self.game_state, 2, 8)
    Q_max, A_max = F.max_q(self.weights)

    # Multiple actions may give the same (optimal) reward. To avoid bias towards
    # a particular action, shuffle the greedy actions.
    shuffle(A_max)
    self.greedy_action = A_max[0]
    
    # Save feature extraction of previous state for Q-learning (the
    # next state is available in reward_update).
    self.F_prev = F

    # Select a random action for use in epsilon-greedy or random-walk
    # exploration. TODO: set actions as global variable
    self.random_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])

    # TODO: Move to reward_update (to allow SARSA use), only use exploitation
    # strategy (greedy) here for known weights

    # Choose a policy depending on the training parameters.
    if t_policy == 'greedy':
        self.logger.info('Pick greedy action %s', self.greedy_action)
        self.next_action = self.greedy_action
    elif t_policy == 'epsgreedy':
        self.logger.info('Picking greedy action at probability %s', 1-t_policy_eps)
        self.next_action = np.random.choice([self.greedy_action,
                                             self.random_action], p=[1-t_policy_eps, t_policy_eps])
    elif t_policy == 'diminishing':
        eps_dim = min(0.05, t_policy_eps * (1 - self.current_game / total_games))
        self.logger.info('Picking greedy action at probability %s', eps_dim)
        self.next_action = np.random.choice([self.greedy_action,
                                             self.random_action], p=eps_dim)
    else:
        raise AttributeError("Invalid policy, check MRBOMBASTIC_POLICY")


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
    # TODO: Use this function to set a marker for training mode

    # We need to perform at least one action before computing an
    # intermediary reward.
    if self.game_state['step'] == 1:
        return None

    # An action is always punished with a small negative reward due to
    # time constraints on the game.
    reward = -1

    for event in self.events:
        if event == e.BOMB_DROPPED:
            # We give no special reward to the action of dropping a
            # bomb, to leave flexibility in the chosen strategies.
            reward += 1
        elif event == e.COIN_COLLECTED:
            # Collecting coins is the secondary goal of the game.
            reward += 100 # 100?
        elif event == e.KILLED_SELF:
            # Killing ourselves through bomb placement is something we
            # wish to avoid.
            reward -= 100
        elif event == e.CRATE_DESTROYED:
            # A bomb may destroy many crates simulatenously; adjust
            # the reward for destroying a single crate accordingly.
            reward += 10
        elif event == e.COIN_FOUND:
            # Give a partial reward for destroying a crate which
            # contains a coin.
            reward += 50 # 30 or 100?
        elif event == e.KILLED_OPPONENT:
            # Killing opponents is the primary goal of the game.
            reward += 300
        elif event == e.GOT_KILLED:
            # Dying at the hands of an opponent is classified worse as
            # death by self.
            reward -= 300
        elif event == e.SURVIVED_ROUND:
            # TODO: Change this to 50?
            reward += 100
        elif event == e.WAITED or e.INVALID_ACTION:
            # An invalid action (such as bumping into a wall) is
            # equivalent to performing no action at all. Punish both
            # with a small negative reward.
            reward -= 2

    self.logger.info('Given reward of %s', reward)
    self.accumulated_reward += reward

    # TODO: update weights either per step, or every "mini batch"
    # Update temporary weights used for behavior policy in each step.
    #self.weights_episode = 

    # Get action performed in last step. This variable was set in the previous 'act'
    # step, and is thus called 'next_action'. The previous state was set in
    # update_reward.
    self.prev_action = self.next_action
    #self.F_prev
    # TODO: set crate/coin_limit in global variable
    # TODO: cache values of F
    F = RLFeatureExtraction(self.game_state, 2, 8)

    # Set learning schedule (alpha) depending on training parameters.
    self.alpha = None
    if t_learning_schedule == 'interval':
        self.alpha = learning_schedule_1(self.game_state['step']-1)
    elif t_learning_schedule == 'quotient1':
        self.alpha = learning_schedule_2(self.game_state['step']-1)
    elif t_learning_schedule == 'quotient2':
        self.alpha = learning_schedule_3(self.game_state['step']-1)
    elif t_learning_schedule == 'fixed':
        self.alpha = learning_schedule_4()
    else:
        raise AttributeError("Invalid learning schedule, check MRBOMBASTIC_ALPHA")

    # # Set learning algorithm (Q/SARSA) depending on learning parameters.
    print("ALPHA:", self.alpha)
    if t_learning_algorithm == 'qlearning':
        self.weights = UpdateWeightsLFA(self.F_prev, self.prev_action, reward, F,
                                        self.weights, self.alpha, self.discount)
        print(self.weights, sep=" ")
    elif t_learning_algorithm == 'sarsa':
        Q_max, A_max = F.max_q(self.weights)
        shuffle(A_max)
        self.greedy_action = A_max[0]
        self.random_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT'])
        self.next_action = np.random.choice([self.greedy_action,
                                             self.random_action], p=[1-t_policy_eps, t_policy_eps])
        self.weights, self.z = UpdateWeightsLFA_SARSA(self.F_prev, self.prev_action, reward, F,
                                                      self.next_action, self.weights, self.z,
                                                      0.2, self.alpha, self.discount)
        #print(self.weights)
    elif t_learning_algorithm == 'doubleq':
        pass # TODO

    # We keep track of all experiences in the episode
    self.replay_buffer += (self.F_prev, self.prev_action, reward, F)
    print("Accumulated reward:", self.accumulated_reward)


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    # TODO: Write accumulated reward to files depending on training_id
    print("Accumulated reward:", self.accumulated_reward)
    self.accumulated_reward = 0
    # TODO: update weights at end of game (10 rounds) to reduce bias (temporary w')
    np.save(weights_file, self.weights)


