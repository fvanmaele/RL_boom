import numpy as np
import pickle
from settings import e
from settings import s
from random import shuffle
from agent_code.my_agent.algorithms import feature_extraction, new_reward, q_gd_linapprox



#########################################################################

def setup(self):
    
    self.actions = [ 'UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT' ]
    self.train_mode = "n_step_EpsGreed"
    self.init_mode = "initX"
    # load weights
    try:
        self.weights = np.load('./agent_code/my_agent/models/weights_{}_{}.npy'.format(self.train_mode, self.init_mode))
        self.training_weights = np.load('train_weights_{}_{}.npy'.format(self.train_mode, self.init_mode))
        self.training_rewards = np.load('train_rewards_{}_{}.npy'.format(self.train_mode, self.init_mode))
        print("weights loaded")
    except:
        self.weights = []
        self.training_weights = []
        self.training_rewards = []
        print("no weights found ---> create new weights")

    # Define Rewards
    self.total_R = 0
    
    # Step size or gradient descent 
    self.n = 10
    self.alpha = 0.2 
    self.gamma = 0.95
    self.EPSILON = 0.2

#####################################################################

def act(self):
    
    """
    actions order: 'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'    
    """
    
    # load state 
    game_state = self.game_state  # isn't it memory waste calling in each feature extraction for coins, self, arena?

    # Compute features state 
    feature_state = feature_extraction(game_state)
    self.prev_state = feature_state

    #different initial guesses can be defined here: 
    if len(self.weights) == 0:
        print('no weights, init weights')
        if self.init_mode == 'initX':
            self.weights = np.array([1,1,-7,-1,4,-0.5,1.5,2,0.5,0.5,-7,1.5,3,2,-1])  
        elif self.init_mode == 'init1':
            self.weights = np.ones(feature_state.shape[1])  
        elif self.init_mode == 'initRand':
            self.weights = np.random.rand(self.prev_state.shape[1])
    
    print(self.weights)

    self.logger.info('Pick action')

    # Linear approximation approach
    
    ####### EPSILON GREEDY (TRAINING) #########################
    
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
    
#####################################################################
def reward_update(self):

    if self.game_state['step']%self.n == 0:

        self.logger.info('IN TRAINING MODE ')
        print('Update Weights')


        reward = new_reward(self.events)
        self.total_R += reward        

        feature_state = feature_extraction(self.game_state)
        next_state = feature_state

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
    if self.game_state['step']%self.n == 0:
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
    np.save('./agent_code/my_agent/models/weights_{}_{}.npy'.format(self.train_mode, self.init_mode), self.weights)

    self.training_weights = np.append(self.training_weights, self.weights)
    np.save('train_weights_{}_{}.npy'.format(self.train_mode, self.init_mode), self.training_weights)
    
    self.training_rewards.append(self.total_R)
    np.save('train_rewards_{}_{}.npy'.format(self.train_mode, self.init_mode), self.training_rewards)
    

