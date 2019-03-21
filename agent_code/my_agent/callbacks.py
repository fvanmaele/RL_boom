import numpy as np
from random import shuffle
from settings import e
from settings import s
from agent_code.my_agent.algorithms import *
import pickle

#########################################################################

def setup(self):
     
    # load weights
    try:

        self.weights = np.load('./agent_code/my_agent/models/weights_initRand')
        self.training_weights = np.load('train_weights_initRand_GreedA02Y095.npy')
        print("weights loaded")
    except:
        self.weights = []
        self.training_weights = []
        print("no weights found ---> create new weights")

    # Define Rewards
    self.total_R = 0
    
    # Step size or gradient descent 
    self.alpha = 0.2 
    self.gamma = 0.95
    self.EPSILON = 0.25
    self.round = 1

#####################################################################

def act(self):
    
    """
    actions order: 'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'    
    """
    
    # load state 
    game_state = self.game_state  # isn't it memory waste calling in each feature extraction for coins, self, arena?
    
    # create BOMB-MAP 
    #ombs = game_state['bombs']
    #arena = game_state['arena']
    #bomb_xys = [(x,y) for (x,y,t) in bombs]
    #bomb_map = np.ones(arena.shape) * 5
    #for xb,yb,t in bombs:
    #    for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) #for h in range(-3,4)]:
    #        if (0 < i < bomb_map.shape[0]) and (0 < j < #bomb_map.shape[1]):
    #            bomb_map[i,j] = min(bomb_map[i,j], t)
    
    # Compute features state 
    f0 = np.ones(6)  # for bias
    f1 = feature1(game_state) # reward good action
    f2 = feature2(game_state) # penalization bad action
    f3 = feature3(game_state)
    f4 = feature4(game_state) # reward good action
    f5 = feature5(game_state)  # penalize bad action
    f6 = feature6(game_state)  # reward good action
    f7 = feature7(game_state) # reward action
#    f8 = feature8(game_state) # rewards good action
    f9 = feature9(game_state) # rewards good action
    f10 = feature10(game_state) # rewards good action
    f11 = feature11(game_state) # penalize bad action
    f12 = feature12(game_state) # penalize bad action
    f13 = feature13(game_state) # penalize bad action
    f14 = feature14(game_state) # penalize bad action
    f15 = feature15(game_state) # penalize bad action
    #feature_state = np.vstack((f0,f1,f2,f4,f5,f6,f7,f8,f9,f10)).T
    feature_state = np.vstack((f0,f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12,f13,f14,f15)).T
   
    self.prev_state = feature_state
    
    #later no necessary
    if len(self.weights) == 0:
        print('no weights, init weights')
        #self.weights = np.array([1,1,-7,-1,4,-0.5,1.5,2,0.5,0.5,-7,1.5,3,2,-1])   #initial 
        #self.weights = np.ones(feature_state.shape[1])  
        self.weights = np.random.rand(feature_state.shape[1])
        #self.weights = weights
    
    print(self.weights)



    self.logger.info('Pick action')
    
    #'''
    
    # Linear approximation approach
    q_approx = np.dot(feature_state, self.weights)    
    best_actions = np.where(q_approx == np.max(q_approx))[0] 
    shuffle(best_actions)

    q_next_action = s.actions[best_actions[0]] #GREEDY POLICY
    self.next_action = q_next_action
    print("q action picked  ", q_next_action)
    #'''
    
    ####### EPSILON GREEDY (TRAINING) #########################
    '''
    greedy = np.random.choice([0,1], p=[self.EPSILON, 1-self.EPSILON])
    if greedy:
    
        q_approx = np.dot(feature_state, self.weights)
        best_actions = np.where(q_approx == np.max(q_approx))[0] 
        shuffle(best_actions)
        
        q_next_action = s.actions[best_actions[0]] #GREEDY POLICY
        self.next_action = q_next_action
        print("q action picked  ", q_next_action)

    else:
        self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
        print("random action picked ", self.next_action)
    '''
    
    
    
    
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
    
    print('LEARNING')

    reward = -1
    for event in self.events:
        if event == e.BOMB_DROPPED:
            reward += 1
        elif event == e.COIN_COLLECTED:
            reward += 200
        elif event == e.KILLED_SELF:
            reward -= 100
        elif event == e.CRATE_DESTROYED:
            reward += 10
        elif event == e.COIN_FOUND:
            reward += 100
        elif event == e.KILLED_OPPONENT:
            reward += 300
        elif event == e.GOT_KILLED:
            reward -= 300
        elif event == e.SURVIVED_ROUND:
            reward += 100
        elif event == e.WAITED or e.INVALID_ACTION:
            reward -= 2

        
    self.total_R += reward
   

    # learning 
    #bombs = self.game_state['bombs']
    #arena = self.game_state['arena']
    #bomb_xys = [(x,y) for (x,y,t) in bombs]
    #bomb_map = np.ones(arena.shape) * 5
    #for xb,yb,t in bombs:
    #    for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
    #        if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
    #            bomb_map[i,j] = min(bomb_map[i,j], t)
                
                
    # Berechene alle features wie in act
    f0 = np.ones(6)  # for bias
    f1 = feature1(self.game_state) # reward good action
    f2 = feature2(self.game_state) # penalization bad action
    f3 = feature4(self.game_state)
    f4 = feature4(self.game_state) # reward good action
    f5 = feature5(self.game_state)  # penalize bad action
    f6 = feature6(self.game_state)  # reward good action
    f7 = feature7(self.game_state) # reward action
    #f8 = feature8(self.game_state) # rewards good action
    f9 = feature9(self.game_state) # rewards good action
    f10 = feature10(self.game_state) # rewards good action
    f11 = feature11(self.game_state)
    f12 = feature12(self.game_state)
    f13 = feature13(self.game_state)
    f14 = feature14(self.game_state)
    f15 = feature15(self.game_state)
    next_state = np.vstack((f0,f1,f2,f3,f4,f5,f6,f7,f9,f10,f11,f12,f13,f14,f15)).T

    if self.game_state['step'] > 1:

        prev_state_a = self.prev_state[s.actions.index(self.next_action),:]

        # update weights
        weights = q_gd_linapprox(next_state, prev_state_a, reward, self.weights, self.alpha, self.gamma)      
        self.weights = weights        
        
        # update alpha and gamma for convergence
        self.alpha = 1/self.game_state['step']
        #self.gamma = self.gamma ** self.game_state['step']
        # update epsilon ... TODO
        

def end_of_episode(self):
    self.round += 1
    #np.save('./agent_code/my_agent/models/weights_initRand.npy', self.weights)
    self.training_weights = np.append(self.training_weights, self.weights)
    print(len(self.training_weights))
    #np.save('train_weights_initRand_GreedA02Y095.npy', self.training_weights)
    
def q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma):
    next_state_a = next_state[np.argmax(np.dot(next_state, weights)), :]
    weights += alpha * (reward + gamma * np.dot(next_state_a,weights) - np.dot(prev_state_a,weights)) * prev_state_a 
    return weights
            
#def eps_greedy(self, epsilon = ):
#    return EPSILON/(C*self.round/s.n_rounds+1)
