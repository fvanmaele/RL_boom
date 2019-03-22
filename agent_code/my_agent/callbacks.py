import numpy as np

from agent_code.my_agent.feature_extraction import *
from agent_code.my_agent.learning_methods import *
from settings import e


def setup(self):
    """Called once before a set of games to initialize data structures etc.

    The 'self' object passed to this method will be the same in all other
    callback methods. You can assign new properties (like bomb_history below)
    here or later on and they will be persistent even across multiple games.
    You can also use the self.logger object at any time to write to the log
    file for debugging (see https://docs.python.org/3.7/library/logging.html).
    """
    np.random.seed()
    self.accumulated_reward = 0 # reward for full game (10 rounds)
    self.reward = 0 # reward for single transition
    self.replay_buffer = [] # tuples (S,A,R,S') for later sampling

    try:
        # load weights
        self.weights = np.load('./agent_code/my_agent/models/weights.npy')
        print("weights loaded")
    except:
        # Initial guess
        self.weights = np.array([1, 1.5, -7, -1, 4, -0.5, 1.5, 1, 0.5, 0.5, 0.8, 0.5])


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

    # An action is always punished with a small negative reward due to
    # time constraints on the game.
    self.reward = -1

    for event in self.events:
        if event == e.BOMB_DROPPED:
            # We give no special reward to the action of dropping a
            # bomb, to leave flexibility in the chosen strategies.
            self.reward += 1
        elif event == e.COIN_COLLECTED:
            # Collecting coins is the secondary goal of the game.
            self.reward += 100
        elif event == e.KILLED_SELF:
            # Killing ourselves through bomb placement is something we
            # wish to avoid.
            self.reward -= 100
        elif event == e.CRATE_DESTROYED:
            # A bomb may destroy many crates simulatenously; adjust
            # the reward for destroying a single crate accordingly.
            self.reward += 10
        elif event == e.COIN_FOUND:
            # Give a partial reward for destroying a crate which
            # contains a coin.
            self.reward += 30
        elif event == e.KILLED_OPPONENT:
            # Killing opponents is the primary goal of the game.
            self.reward += 300
        elif event == e.GOT_KILLED:
            # Dying at the hands of an opponent is classified worse as
            # death by self.
            self.reward -= 300
        elif event == e.SURVIVED_ROUND:
            # TODO: Change this to 50?
            self.reward += 100
        elif event == e.WAITED or e.INVALID_ACTION:
            # An invalid action (such as bumping into a wall) is
            # equivalent to performing no action at all. Punish both
            # with a small negative reward.
            self.reward -= 2
    
    # We keep track of all experiences in the episode
    self.logger.info(f'Given reward of {reward}')
    self.accumulated_reward += reward
    self.replay_buffer += (self.prev_state, self.
    
    # Get action performed in last step.
    self.prev_action = self.next_action # self.prev_state

    # TODO: update weights either per step, or every "mini batch"
    self.weights = QLearningLinFApp(self.prev_state, self.prev_action, self.reward,
                                    self.game_state, self.weights, feature, 0.05, 1)


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
    # Save previous state for Q-learning in reward_update.
    self.prev_state = self.game_state

    #self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'], p=[.25, .25, .25, .25])

    # TODO: Implement Q-policy here


def end_of_episode(self):
    """Called at the end of each game to hand out final rewards and do training.

    This is similar to reward_update, except it is only called at the end of a
    game. self.events will contain all events that occured during your agent's
    final step. You should place your actual learning code in this method.
    """
    np.save('./agent_code/my_agent/models/weights.npy', self.weights)
