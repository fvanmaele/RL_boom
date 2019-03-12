import numpy as np
import pickle
import os
import algorithm.QLearningLinApprox.LinearApproxQLearning as ql 


def setup(self):
    np.random.seed()
    
    #load w from file if file exists, or create w 
    if os.path.isfile('w.pickle'):
        file = open('w.pickle', 'rb')
        self.w = pickle.load(file)
        file.close()
    else:
        '''
        initiate w with vector size n of 1s
        n should be dependent of f (vector multiplication)
        '''
        self.w = np.fill(n,1)
    
    
    

def act(self):

    self.logger.info('Pick action using q learning')
    
    #save previous state in self.prev_state for learning
    self.prev_state = self.game_state
    
    '''
    use q learning  with linear approximation to calculate next action
    TODO: implement exploration (do random actions every now and then)
    
    self.next_action is the action taken after the algorithm
    calculates what the next action should be.
    '''
    self.next_action, self.q = ql.pred(self.game_state, self.w, f)
    

def reward_update(self):
   
    '''
        IMPORTANT: 
            reward_update happens AFTER act
            This means, self.next_action in reward_update is the
            action the algorithm just took in action. This is why
            we should rename self.next_action as prev_action in 
            reward_update
        
        
        REWARDS -> Ferdinand
    '''
    #update weights after each step
    prev_action = self.next_action
    
    self.w = QLearn_Lin_Approx(self.prev_state, prev_action, self.game_state, reward, w, f)
    
    #self.w, = QLearn_Lin_Approx(self.prev_state, prev_action, self.q, reward, w, f)

def end_of_episode(self):
    
    #save learned w in 'w.pickle' after each episode
    file = open('w.pickle', 'wb')
    pickle.dump(self.w, file)
    file.close()
    
    
