
import numpy as np
from random import shuffle
from settings import e
from settings import s
from agent_code.my_agent.feature_extraction import *
#from agent_code.my_agent.learning_methods import *


def setup(self):
    # Assumed order for features (algorithms.py)
    self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

    # load weights
    try:
        self.weights = np.load('./agent_code/my_agent/models/weights.npy')
        print("weights loaded")
    except:
        self.weights = []
        print("no weights found ---> create new weights")

    # Define Rewards
    self.total_R = 0

    # Step size or gradient descent 
    self.alpha = 0.2 
    self.gamma = 0.9
    self.EPSILON = 0.2


def act(self):
    # load state 
    game_state = self.game_state  # isn't it memory waste calling in each feature extraction for coins, self, arena?
    print("\n", game_state['step'])

    F = RLFeatureExtraction(game_state)
    feature_state = F()

    self.prev_state = feature_state
    weights = np.array([1, 1.5, -7, -1, 4, -0.5, 1.5, 1, 0.5, 0.5, 0.8, 0.5])   #initial guess 
    print("weights", weights)
#    # later no necessary
#    if self.weights == []:
#        print(feature_state.shape[0])
#        weights = np.ones(feature_state.shape[1])  
#        self.weights = weights
#    else:
#        weights = self.weights
#

    # Linear approximation approach
    q_approx = linapprox_q(feature_state, weights)
    print("q", q_approx)
    best_actions = np.where(q_approx == np.max(q_approx))[0] 
    shuffle(best_actions)
    q_next_action = self.actions[best_actions[0]] #GREEDY POLICY

    self.next_action = q_next_action
    print(self.next_action)


def reward_update(self):
    """IMPORTANT:

    reward_update happens AFTER act This means, self.next_action in
    reward_update is the action the algorithm just took in
    action. This is why we should rename self.next_action as
    prev_action in reward_update
    """
    self.logger.info('IN TRAINING MODE ')

    reward = 0 

    if self.game_state['step'] != 1:
        for event in self.events:
            if event == e.INVALID_ACTION:
                reward -= 10 
            elif event == e.COIN_COLLECTED:
                reward += 100
            elif event == e.WAITED:
                reward -= 10 
            else:
                reward -= 1
        
        self.total_R += reward
       
        # load weights
        weights = self.weights

        # learning 
    
        # Berechene alle features
        """
        so wie vorhin und mit stack, vlt, machen wir eine Function die alles auf einmal stack
        """
        prev_state = self.prev_state
        prev_state_a = prev_state[s.actions.index(self.next_action),:]

        alpha = self.alpha
        gamma = self.gamma
        # update weights
        weights = q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma)        

        self.weights = weights
        self.alpha = 1/self.game_state['step']
        self.gamma = self.gamma ** self.game_state['step']
        

def end_of_episode(self):
    np.save('./agent_code/my_agent/models/weights.npy', self.weights)
    

def linapprox_q(state, weights):
    q_approx = np.dot(state, weights)
    return q_approx

def q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma):
    next_state_a = next_state[np.argmax(linapprox_q(next_state, weights)), :]
    weights += alpha * (reward + gamma * np.dot(next_state_a,weights) - np.dot(prev_state_a,weights)) * prev_state_a 
    return weights
            
#def eps_greedy(self, epsilon = ):
#    return EPSILON/(C*self.round/s.n_rounds+1)
