
import numpy as np
from random import shuffle
from settings import e
from settings import s
from agent_code.my_agent.algorithms import *
#from agent_code.my_agent.features import *
#from agent_code.my_agent.functions import *

def tmp_feature4(game_state):
    '''
    This feature rewards the action that minimizes the distance to safety
    should the agent be in the danger zone(where explosions will be).
        F(s,a) = 1, should a reduces distance to safety
        F(s,a) = 0, otherwise
    F(s,a) returns 0 if we are not in the danger zone   
    
    We begin by extracting all relevant information from game_state
    '''
    x, y, _, bombs_left = game_state['self']
    arena = game_state['arena']
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]

    '''
    and initializing the resulting vector
    '''
    features = []

    danger_zone = [] 
    if len(bombs) != 0:
        for b in bombs_xy:
            danger_zone += compute_blast_coords(arena, b)
    else:
        return np.array([0,0,0,0,0,0])

    danger_zone += bombs_xy

    for d in directions:
        if ((arena[d] != 0) or (d in others) or (d in bombs_xy)):
            d = (x,y)
        if d in danger_zone:
            j
        arena[danger_zone] = 5       
        target = np.argwhere(arena==0)
    #d = look_for_targets(free_space, (agent[0], agent[1]), safe_loc)
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

    # Step size or gradient descent 
    self.alpha = 0.2 
    self.gamma = 0.9
    self.EPSILON = 0.2

def act(self):

    """
    actions order: 'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'    
    """

    # load state 
    game_state = self.game_state  # isn't it memory waste calling in each feature extraction for coins, self, arena?
    print("\n", game_state['step'])

    # create BOMB-MAP 
    bombs = game_state['bombs']
    arena = game_state['arena']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)
    
    # Compute features state 
    f0 = np.ones(6)  # for bias
    print("f0 ", f0)
    f1 = feature1(game_state) # reward good action
    print("f1 ", f1)
    f2 = feature2(game_state) # penalization bad action
    print("f2 ", f2)
    f3 = feature3(game_state)
    print("f3 ", f3)
    f4 =feature4(game_state) # reward good action
    print("f4 ", f4)
    f5 = feature5(game_state)  # penalize bad action
    print("f5 ", f5)
    f6 = feature6(game_state)  # reward good action
    print("f6 ", f6)
    f7 = feature7(game_state) # reward action
    print("f7 ", f7)
#    f8 = feature8(game_state) # rewards good action
#    print("f8 ", f8)
    f9 = feature9(game_state) # rewards good action
    print("f9 ", f9)
    f10 = feature10(game_state) # rewards good action
    print("f10 ", f10)
    f11 = feature11(game_state) # penalize bad action
    print("f11 ", f11)
    f12 = feature12(game_state) # penalize bad action
    print("f12 ", f12)
    f13 = feature13(game_state) # penalize bad action
    print("f13 ", f13)
    f14 = feature14(game_state) # penalize bad action
    print("f14 ", f14)
    f15 = feature15(game_state) # penalize bad action
    print("f15 ", f15)

    #feature_state = np.vstack((f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10)).T
    feature_state = np.vstack((f0,f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12,f13,f14,f15)).T


    self.prev_state = feature_state
    weights = np.array([1,1,-7,-1,4,-0.5,1.5,2,0.5,0.5,-7,1.5,3,2,-1])   #initial guess 
    #weights = np.array([1,1,-5,4,-0.5,1,1])   #initial guess 
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
    q_next_action = s.actions[best_actions[0]] #GREEDY POLICY
#
#    # Pick action
#    self.logger.info('Pick action')

#    self.next_action = np.random.choice(['WAIT','RIGHT', 'LEFT', 'DOWN', 'UP','BOMB'], p=[0.15, .15, 0.15, .15, 0.15, 0.25])

#    print("action picked  ", q_next_action)
    self.next_action = q_next_action
    print(self.next_action)

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
