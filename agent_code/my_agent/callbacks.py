import numpy as np
import pickle
from settings import e
from settings import s
from random import shuffle
from agent_code.my_agent.feature_extraction import *
from agent_code.my_agent.algorithms import feature_extraction,new_reward, q_gd_linapprox


#########################################################################

def setup(self):
    
    self.actions = [ 'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT' ]
    self.init_mode = 'initX'
    # Define Rewards
    self.total_R = 0

    # Step size or gradient descent 
    self.alpha = 0.2
    self.gamma = 0.95
    self.EPSILON = 0.2
    self.round = 1
    
    # load weights
    try:
        self.weights = np.load('./training_res/test_2.npy')
        print("weights loaded")
    except:
        self.weights = []
        print("no weights found ---> create new weights")



#####################################################################

def act(self):
    
    """
    actions order: 'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'    
    """

    # Compute features state 
    #'''
    F = RLFeatureExtraction(self.game_state)
    feature_state1 = F.state()
    #self.prev_state = feature_state1
    #'''
    #'''
    feature_state = feature_extraction(self.game_state)
    self.prev_state = feature_state
    #'''
    
    
    
    #different initial guesses can be defined here: 
    if len(self.weights) == 0:
        print('no weights, init weights')
        if self.init_mode == 'initX':
            self.weights = np.array([1,1,-7,-1,4,-0.5,1.5,2,0.5,0.5,-7,1.5,3,2,-1])
            #self.weights = np.array([1,1,-7,-1,4,-0.5,1.5,2,0.5,0.5,-7,1.5,3])            
        elif self.init_mode == 'init1':
            self.weights = np.ones(feature_state.shape[1])  
        elif self.init_mode == 'initRand':
            self.weights = np.random.rand(self.prev_state.shape[1])
    
    print(self.weights)
    print("alpha: ",self.alpha)
    print('feature_state_1',feature_state1)
    print('feature_state_0',feature_state)
    self.logger.info('Pick action')
    
    #'''
    # Linear approximation approach
    q_approx = np.dot(feature_state, self.weights)    
    best_actions = np.where(q_approx == np.max(q_approx))[0] 
    shuffle(best_actions)
    print(best_actions)
    q_next_action = self.actions[best_actions[0]] #GREEDY POLICY
    self.next_action = q_next_action
    print("q action picked  ", self.next_action)
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
#####################################################################
def reward_update(self):

    self.logger.info('IN TRAINING MODE ')
    print('LEARNING')

    print('EVENTS: ',self.events)
    reward = new_reward(self.events)
    print('Rewards: {}'.format(reward))
    self.total_R += reward        
    
    '''
    F = RLFeatureExtraction(self.game_state)
    next_state = F.state()
    '''
    
    next_state = feature_extraction(self.game_state)
    
    if self.game_state['step'] > 1:

        prev_state_a = self.prev_state[self.actions.index(self.next_action),:]

        # update weights
        weights = q_gd_linapprox(next_state, prev_state_a, reward, self.weights, self.alpha, self.gamma)      
        self.weights = weights        
        
        # update alpha and gamma for convergence
        self.alpha = 0.2/self.game_state['step']
        #self.gamma = self.gamma ** self.game_state['step']
        
#####################################################################
def end_of_episode(self):

    self.alpha = 0.2
    ## calculate new weights for last step
    reward = new_reward(self.events)
    self.total_R += reward        
    
    feature_state = feature_extraction(self.game_state)
    next_state = feature_state

    prev_state_a = self.prev_state[self.actions.index(self.next_action),:]

    # update weights
    weights = q_gd_linapprox(next_state, prev_state_a, reward, self.weights, self.alpha, self.gamma)      
    self.weights = weights 

    ############## SAVING LEARNING FROM ONE EPISODE 
    #np.save('./agent_code/my_agent/models/NONE.npy', self.weights)

