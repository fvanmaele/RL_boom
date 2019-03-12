
import numpy as np
from random import shuffle

from settings import e

def feature_extraction_tmp(self):
    coins = self.game_state['coins']
    x, y, _, bombs_left = self.game_state['self']
    feat_coins = np.sum( np.abs( np.asarray(coins) - np.array([x,y]) ), axis = 1)
    print(feat_coins)
    return feat_coins

def feature_extraction(coins, arena, directions):
    """
    This function is just implemented for the coin dectection
    """
    feat_coins = [] 
    for d in directions:
        if arena[d] == 0:
            diff2coins = np.abs( np.asarray(coins) - np.array([d[0],d[1]]) )
            for diff in diff2coins:
                if diff[0] == 0 or diff[1] == 0:
                    

            # 28 we could change this value
            
            feat_coins.append(28 - min(np.sum( np.abs( np.asarray(coins) - np.array([d[0],d[1]]) ), axis = 1)))
        else:
            feat_coins.append(0)
    print(feat_coins)
    #coins = self.game_state['coins']
    #x, y, _, bombs_left = self.game_state['self']
    #feat_coins = np.sum( np.abs( np.asarray(coins) - np.array([x,y]) ), axis = 1)
    #print(feat_coins)
    return np.asarray(feat_coins)

def setup(self):
    np.random.seed()
    #self.weights = np.load("./weights.npy") 
    
def act(self):
    self.logger.info('Picking cool action set')
    
    # for coin detection
    coins = self.game_state['coins']
    x, y, _, bombs_left = self.game_state['self']
    arena = self.game_state['arena']

    directions = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    feat_coins = feature_extraction(coins, arena, directions)
    
    self.logger.info('Pick action ')

    actions = ['RIGHT', 'LEFT', 'DOWN', 'UP']

    best_action = actions[feat_coins.argmax()]
    self.next_action = np.random.choice([best_action, 'RIGHT', 'LEFT', 'UP', 'DOWN'], p=[.8, .05, .05, .05, .05])
    print(best_action)
#    self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])
#    self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB'], p=[.23, .23, .23, .23, .08])

def reward_update(self):
#    coins = self.game_state['coins']
#    x, y, _, bombs_left = self.game_state['self']
#    arena = self.game_state['arena']
#    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)]
#    feature_extraction(coins, arena, directions)
#    np.save("./weights.npy", weights)

    # Compute acumulative rewards
    # TODO:  CHANGE THIS VALUES (THIS ARE FROM WAPU)
    #for event in self.game_state['events']:
    #    elif event == e.COIN_COLLECTED:
    #        reward += 100
    #    elif event == e.WAITED:
    #        reward -= 2
    #    elif event == e.INVALID_ACTION:
    #        reward -= 2

    pass
def end_of_episode(agent):
    pass
