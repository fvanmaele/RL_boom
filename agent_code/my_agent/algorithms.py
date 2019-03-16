import numpy as np
from random import shuffle

############################### Useful functions for differrent computations

def compute_patch(arena, p1, p2):
    """
    this function computes the patch of the arena between the points p1 and p2
    """
    patch = arena[min(p1[1], p2[1]):max(p1[1], p2[1])+1,  min(p1[0], p2[0]):max(p1[0], p2[0])+1] 
    return patch


############################### Features

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
