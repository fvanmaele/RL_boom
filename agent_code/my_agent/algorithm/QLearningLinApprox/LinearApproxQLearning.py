import numpy as np

'''
Q Learning using linear approximation, with feature selection f as a function
for modularity 

all possible actions are saved here for simplicity as a global parameter.
'''

actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']

''' 
def update_w(state, action, next_state, reward, w, f, step_size=0.05, discount=1):
    vec_state = f(state, action)
    delta = 0
    
    max_q = 0
    
    for a in actions:
        q = w.T*f(next_state, a)
        if max_q < q:
            max_q = q
        
    delta = reward + discount*max_q - (w.T*f(state,action))
    
    return w+step_size * delta * f(state,action)

'''
def update_w_greedy(state, action, q, reward, w, f, step_size=0.05, discount=1):
    vec_state = f(state, action)
        
    delta = reward + discount*q - np.dot(w.T*f(state,action))
    
    return w + step_size * delta * f(state,action)
   

def pred(game_state, w, f):

    '''
    returns the action with the highest q value calculated
    using the linear approximation of q with vector w.
    
    default value of next_action is 'WAIT', hence best_a 
    is also set to 'WAIT' in the beginning
    '''
    best_q = 0
    best_a = 'WAIT'
    
    for a in actions:
        q = np.dot(w.T*f(game_state, a))
        if max_q<q:
            best_q = q
            best_a = a
    
    return best_a, best_q
    

