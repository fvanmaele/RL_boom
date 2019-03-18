
import numpy as np
from random import shuffle
from time import time, sleep
from collections import deque
import pickle
import copy

from settings import s
from settings import e


############# FEATURES ##############

def feature1(game_state):
    """
    Reward the best possible action to a coin, if it is reachable F(s,a)=1,  otherwise F(s,a)=0.    
    `BOMB' and `WAIT ' are always 0
    """
    coins = game_state['coins']
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']

    feature = [] # Desired feature
    
    # construct the free_space Boolean numpy_array
    free_space = arena == 0

    best_coord = look_for_targets(free_space, (x,y), coins)

    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]

    if best_coord is None:
        return np.zeros(6)
    for d in directions:
        if d != best_coord:
            feature.append(0)
        else:
            feature.append(1)
    
   # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)
    
    return np.asarray(feature)


def feature2(game_state, bomb_map):
    """
    Penalize if the action follows the agent to death (F,s)=1, F(s,a)=0. otherwise.
    it could be maybe get better
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    explosions = game_state['explosions'] 
    arena = game_state['arena']
    bomb_power = s.bomb_power
    blast_coords = bombs_xy 
    feature = []
    
    danger_zone = [] 
    if len(bombs) != 0:
        '''
        If there are bombs on the game board, we map all the explosions
        that will be caused by the bombs. For this, we use the help function
        get_blast_coords. This is an adjusted version of the function with the
        same name from item.py of the original framework
        '''
        for b in bombs_xy:
            danger_zone += compute_blast_coords(arena, b)
    danger_zone += bombs_xy

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            d = (x,y)
            
        if ((d in danger_zone) and  bomb_map[d] == 0) or explosions[d] ==2:
            feature.append(1) 
        else:
            feature.append(0)
    
    # BOMB actions should be same as WAIT action
    feature.append(feature[-1])

    return np.asarray(feature)


def feature4(game_state):
    '''
    This feature rewards the action that minimizes the distance to safety
    should the agent be in the danger zone(where explosions will be).
        F(s,a) = 1, should a reduces distance to safety
        F(s,a) = 0, otherwise
    F(s,a) returns 0 if we are not in the danger zone   
    
    We begin by extracting all relevant information from game_state
    '''
    agent = game_state['self']
    arena = game_state['arena']
    bombs = game_state['bombs']
    '''
    and initializing the resulting vector
    '''
    res = np.zeros(6, dtype=np.int8)
    if len(bombs) != 0:
        '''
        If there are bombs on the game board, we map all the explosions
        that will be caused by the bombs. For this, we use the help function
        get_blast_coords. This is an adjusted version of the function with the
        same name from item.py of the original framework
        '''
        danger_zone = []
        for b in bombs:
            danger_zone= get_blast_coords(b, arena, danger_zone)

        '''
        We then check if the agent is in the danger zone
        If agent is not in the danger zone, we return 0.
        Otherwise we calculate the distance/direction of safety.
        '''
        if agent[0] in danger_zone[:,0] and agent[1] in danger_zone[:,1]:
            
            '''
            we then mark these explosions on our map. here we deep-copy
            the arena, so that in the case that the arena is needed for
            other features, it remains unchanged
            '''
            map_ = copy.deepcopy(arena)
            map_[danger_zone[:,0], danger_zone[:,1]] = 2
            '''
            '''
            safe_loc = np.argwhere(map_==0)
            free_space = abs(map_) != 1
            d = look_for_targets(free_space, (agent[0], agent[1]), safe_loc)
            '''
            we then calculate the minimum distance of our agent to any of these safe locations.
            For simplicity, only Manhattan distance is used to calculate distance in this
            feature extraction. However, this doesn't always represent the true distance in the 
            game because of the walls.(Possible point of improvement?)
 
            Then, we calculate the positions of our agent after taking all possible actions.
            '''
            
            actions_loc = np.array([(agent[0], agent[1]-1), #up
                                    (agent[0], agent[1]+1), #down
                                    (agent[0]-1, agent[1]), #left
                                    (agent[0]+1, agent[1]), #right
                                    (agent[0], agent[1]),   #bomb
                                    (agent[0], agent[1])])  #wait
            
            res = (actions_loc[:,0] == d[0]) & (actions_loc[:,1] == d[1])
            res = res.astype(int)
     
    return res

def feature5(game_state):
    """
    Penalize invalid actions.  F(s,a) = 1, otherwise F(s,a) = 0.  
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    arena = game_state['arena']

    feature = [] # Desired feature

    for d in directions:
        if ((arena[d] != 0) or 
            (d in others) or 
            (d in bombs_xy)):
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB'
    if bombs_left == 0:
        feature.append(1)
    else:
        feature.append(0)

    # for 'WAIT'
    feature.append(0)

    return np.asarray(feature)

def feature6(game_state):
    """
    Reward when getting a coin F(s,a) = 1, otherwise F(s,a) = 0
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    bombs = game_state['bombs']
    bombs_xy = [(x,y) for (x,y,t) in bombs]
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    coins = game_state['coins']
    arena = game_state['arena']

    feature = [] # Desired feature

    for d in directions:
        if (x,y) in coins:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(0)

    return np.asarray(feature)

def feature7(game_state):
    """
    Reward putting a bomb next to a block. 
    F(s,a) = 0 for all actions and F(s,a) = 1 for a 'BOMB' if we are next to a block.
    """
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    
    feature = [0, 0, 0, 0] # Desired feature

    # add feature for the action 'BOMB'
    # check if we are next to a crate
    CHECK_FOR_CRATE = False 
    for d in directions:
        if arena[d] == 1:
            CHECK_FOR_CRATE = True 
            break
    if CHECK_FOR_CRATE:
        feature.append(1)
    else:
        feature.append(0)

    # add feature for 'WAIT'
    feature.append(0)

    return np.asarray(feature)


def feature8(game_state):
    """
    Reward (if there are no blocks anymore ? and no coins?)  the available movements F(s,a) = 1, 
    otherwise F(s,a) = 0 .   Bombs = 0, WAIT =1 ? 
    """

    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y)]
    arena = game_state['arena']
    
    feature = [] # Desired feature

    for d in directions:
        if arena[d] == 0:
            feature.append(1)
        else:
            feature.append(0)

    # for 'BOMB' and 'WAIT'
    feature.append(0)
    feature.append(1)

    return np.asarray(feature)
