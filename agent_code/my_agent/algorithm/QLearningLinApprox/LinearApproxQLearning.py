import numpy as np

'''
Q Learning using linear approximation, with feature selection f as a function
for modularity 

!!!Implementation details to follow (numpy functions for vector multiplications)!!!
'''
def update_w(state, action, next_state, reward, w, f, step_size=0.05, discount=1):
    vec_state = f(state, action)
    delta = 0
    
    actions = []
    max_q = 0
    #max_action = 0
    
    for a in actions:
        q = w.T*f(next_state, a)
        if max_q < q:
            max_q = q
            #max_action = a
        
    delta = reward + discount*max_q - (w.T*f(state,action))
    
    return w+step_size * delta * f(state,action)
    
    
def train(game_states, )