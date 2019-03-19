import numpy as np
from random import shuffle
from settings import e
from settings import s
from agent_code.my_agent.algorithms import *


#########################################################################

def setup(self):
     
    # load weights
    try:
        self.weights = np.load('./agent_code/my_agent/models/weights_initx')
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


#    # While this timer is positive, agent will not hunt/attack opponents
#    self.ignore_others_timer = 0
#########################################################################

def act(self):

    """
    actions order: 'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'    
    """
    print("############################")
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
    f4 = feature4(game_state) # reward good action
    f5 = feature5(game_state)  # penalize bad action
    f6 = feature6(game_state)  # reward good action
    f7 = feature7(game_state) # reward action
    f8 = feature8(game_state) # rewards good action
    f9 = feature9(game_state) # rewards good action
    f10 = feature10(game_state) # rewards good action
    feature_state = np.vstack((f0,f1,f2,f4,f5,f6,f7,f8,f9,f10)).T
   
    self.prev_state = feature_state
    
    #weights = np.array([1,1,-1,-1,1])   #initial guess 
    #weights = np.array([1,1,-7,5,-0.5,1.5,2,0.5]) 
    #later no necessary
    if len(self.weights) == 0:
        #print('no weights, init weights')
        self.weights = np.array([1,1,-7,4,-0.5,1.5,2,0.5,0.5,0.5])
        #self.weights = np.ones(feature_state.shape[1])  
        #self.weights = weights
    
    print(self.weights)
    #print(self.game_state['step']) 
    #print(feature_state)

    self.logger.info('Pick action')
    
    # Linear approximation approach with epsilon-greedy
    #greedy = np.random.choice([0,1], p=[self.EPSILON, 1-self.EPSILON])
    #if greedy:
    
    q_approx = linapprox_q(feature_state, self.weights)
    best_actions = np.where(q_approx == np.max(q_approx))[0] 
    shuffle(best_actions)
    q_next_action = s.actions[best_actions[0]] #GREEDY POLICY
    self.next_action = q_next_action
    print("q action picked  ", q_next_action)


    #else:
        #self.next_action = np.random.choice(['WAIT','RIGHT', 'LEFT', 'DOWN', 'UP','BOMB'], p=[0.15, .15, 0.15#, .15, 0.15, 0.25])
        #self.next_action = random_action
        #print("random action picked ", self.next_action)
    
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
        bombs = self.game_state['bombs']
        arena = self.game_state['arena']
        bomb_xys = [(x,y) for (x,y,t) in bombs]
        bomb_map = np.ones(arena.shape) * 5
        for xb,yb,t in bombs:
            for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i,j] = min(bomb_map[i,j], t)
                    
                    
        # Berechene alle features
        f0 = np.ones(6)  # for bias
        f1 = feature1(self.game_state) # reward good action
        f2 = feature2(self.game_state) # penalization bad action
        f4 = feature4(self.game_state) # reward good action
        f5 = feature5(self.game_state)  # penalize bad action
        f6 = feature6(self.game_state)  # reward good action
        f7 = feature7(self.game_state) # reward action
        f8 = feature8(self.game_state) # rewards good action
        f9 = feature9(self.game_state) # rewards good action
        f10 = feature10(self.game_state) # rewards good action
        next_state = np.vstack((f0,f1,f2,f4,f5,f6,f7,f8,f9,f10)).T
        
        """
        so wie vorhin und mit stack, vlt, machen wir eine Function die alles auf einmal stack
        """
        prev_state = self.prev_state
        prev_state_a = prev_state[s.actions.index(self.next_action),:]

        alpha = self.alpha
        gamma = self.gamma
        # update weights
        weights = q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma)        
        #print('weights: ', weights)
        #print('weights updated')
        self.weights = weights
        self.alpha = 1/self.game_state['step']
        self.gamma = self.gamma ** self.game_state['step']
        

def end_of_episode(self):
    np.save('./agent_code/my_agent/models/weights_initx', self.weights)
    

def linapprox_q(state, weights):
    #print('shape_state', state.shape, 'shape_weights', weights.shape)
    q_approx = np.dot(state, weights)
    return q_approx

def q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma):
    next_state_a = next_state[np.argmax(linapprox_q(next_state, weights)), :]
    weights += alpha * (reward + gamma * np.dot(next_state_a,weights) - np.dot(prev_state_a,weights)) * prev_state_a 
    return weights
            
#def eps_greedy(self, epsilon = ):
#    return EPSILON/(C*self.round/s.n_rounds+1)
