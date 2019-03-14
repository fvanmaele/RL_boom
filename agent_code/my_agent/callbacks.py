
import numpy as np
from random import shuffle
from settings import e
#from agent_code.my_agent.algorithms import *

def compute_patch(arena, p1, p2):
    """
    this function computes the patch of the arena between the points p1 and p2
    """
    patch = arena[min(p1[1], p2[1]):max(p1[1], p2[1])+1,  min(p1[0], p2[0]):max(p1[0], p2[0])+1] 
    return patch

def feat_1(game_state):
    """
        Feature extraction for coin detection
    """
    coins = game_state['coins']
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)] # define directions
    arena = game_state['arena']

    feature = [] # Desired feature

    for d in directions:
        f = []  # array that helps building the features 
        if arena[d] != 0:
            d = directions[0]   # Don't move if it is an invalid action

        diff2coins = np.abs( np.asarray(coins) - np.array([d[0],d[1]]) )
        sum_diff2coins = np.sum(diff2coins, axis=1)

        # min distance along x & y axis and 'global min distance'
        diffmin_xy = np.min(diff2coins, axis=0)  # min difference of distances
        min_coin, dist2min_coin = np.argmin(sum_diff2coins), np.min(sum_diff2coins)
        
        list_mincoins = list(np.where(sum_diff2coins == dist2min_coin)[0])
        
        # find how many walls are in between (finding the patches between the agent and coin)
        patches = [] 
        for m in list_mincoins:
            p = compute_patch(arena, coins[m], d)
            patches.append(p)

        # not necesary any more because we are computing all the patches
        patch_coinagent = compute_patch(arena, coins[min_coin], d)
        
        # look if there is a fast path to the closes coin
        FAST_PATH = False 
        for patch in patches:
            if patch.shape[0] == 1 or patch.shape[1] == 1:
                if np.count_nonzero(patch) == 0:
                    FAST_PATH=True
                    break
            else:
                FAST_PATH=True
                break
        if not FAST_PATH:
            dist2min_coin += 2

        # fill features
        """
        other posible features
        f.append(diffmin_xy[0])
        f.append(diffmin_xy[1])
        f.append(np.count_nonzero(patch_coinagent))
        f.append(patch_coinagent.shape[0]* patch_coinagent.shape[1] - np.count_nonzero(patch_coinagent))
        """
        f.append(28- dist2min_coin)
        
        feature.append(f)

    feature = np.asarray(feature) 

    # because this feature doesn't take in consideration using bombs
    f_bomb = np.expand_dims(np.zeros(feature.shape[1]), axis=0)
    feature = np.concatenate((feature,f_bomb), axis=0)

    return feature






##################################################################################################################

def setup(self):
    
    # load weights
    try:
        self.weights = np.load('./agent_code/my_agent/models/weights.npy')
        print("weights loaded")
    except:
        self.weights = []
        print("no weights found ---> create new weights")

    # Define Rewards
    self.total_R = 0

    # Define possible actions
    self.actions = ['WAIT','RIGHT', 'LEFT', 'DOWN', 'UP'] #Bomb
    
    # Step size or gradient descent 
    self.alpha = 0.2 
    self.gamma = 0.9

def act(self):

    """
    For the moment only trying to solve the coin detection problem
    actions and  order: 'WAIT','RIGHT', 'LEFT', 'DOWN', 'UP', 'BOMB'  (SEE: setup)
    """

    game_state = self.game_state  # isn't it memory waste calling in each feature extraction for coins, self, arena?

    # Create new feature sates
    f1 = feat_1(game_state)
    """
    Idea would be compute more features:
    f1 = ... 
    ...
    ...
    f = stack ( all features) 
    """
    feature_state = f1
    self.prev_state = feature_state
    print("features computed")

    # later no necessary
    if self.weights == []:
        weights = np.ones(feature_state.shape[1])  
        self.weights = weights
    else:
        weights = self.weights

    # TODO:  implement with shuffle for equally best actions ??
    q_approx = linapprox_q(feature_state, weights)
    q_next_action = self.actions[np.argmax(q_approx)] #GREEDY POLICY

    self.logger.info('Pick action ')
#    self.next_action = np.random.choice(['WAIT','RIGHT', 'LEFT', 'DOWN', 'UP'], p=[0.2, .20, .20, .20, .20])
#    print(game_state['arena'])

    self.next_action = q_next_action
    print("action", q_next_action)
     
        
    print(self.weights)

def reward_update(self):

    '''
        IMPORTANT: 
            reward_update happens AFTER act
            This means, self.next_action in reward_update is the
            action the algorithm just took in action. This is why
            we should rename self.next_action as prev_action in 
            reward_update
        
    '''
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
        next_state = feat_1(self.game_state)
        """
        so wie vorhin und mit stack, vlt, machen wir eine Function die alles auf einmal stack
        """
        prev_state = self.prev_state
        prev_state_a = prev_state[self.actions.index(self.next_action),:]

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
