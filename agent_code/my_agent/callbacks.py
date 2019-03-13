
import numpy as np
from random import shuffle
from settings import e


#def feature_extraction_tmp(self):
#    coins = self.game_state['coins']
#    x, y, _, bombs_left = self.game_state['self']
#    feat_coins = np.sum( np.abs( np.asarray(coins) - np.array([x,y]) ), axis = 1)
#    print(feat_coins)
#    return feat_coins
#
#def feature_extraction(coins, arena, directions):
#    """
#    This function is just implemented for the coin dectection
#    """
#    feat_coins = [] 
#    for d in directions:
#        if arena[d] == 0:
#            diff2coins = np.abs( np.asarray(coins) - np.array([d[0],d[1]]) )
#            for diff in diff2coins:
#                if diff[0] == 0 or diff[1] == 0:
#                    
#
#            # 28 we could change this value
#            
#            feat_coins.append(28 - min(np.sum( np.abs( np.asarray(coins) - np.array([d[0],d[1]]) ), axis = 1)))
#        else:
#            feat_coins.append(0)
#    print(feat_coins)
#    #coins = self.game_state['coins']
#    #x, y, _, bombs_left = self.game_state['self']
#    #feat_coins = np.sum( np.abs( np.asarray(coins) - np.array([x,y]) ), axis = 1)
#    #print(feat_coins)
#    return np.asarray(feat_coins)


#def feat_2(game_state):
#    """
#        Feature extraction for coin detection
#    """
#     
#    coins = game_state['coins']
#    x, y, _, bombs_left = game_state['self']
#    directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)] # define directions
#    arena = game_state['arena']
#
#    max_distance = arena.shape[0]+arena.shape[1] -6 # because of the minus and 'next move'
#
#    f = []
#    for d in directions:
#        if arena[d] == 0:
#            diff2coins = np.abs( np.asarray(coins) - np.array([d[0],d[1]]) )
#        else:
#            
#            diff2coins = np.abs( np.asarray(coins) - np.array([d[0],d[1]]) ) + 1  # avoid special case
#        
#        for diff in diff2coins:
#            if diff[0] == 0 and arena[d] != 0 :
#                
#        min(np.sum(np.abs( np.asarray(coins) - np.array([d[0],d[1]]) ), axis = 1))
#        f.append(min(np.sum(diff2coins, axis = 1)))
#    print(f)
#    return np.asarray(f)
#

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
        # min distance along x & y axis and 'global min distance'
        diffmin_xy = np.min(diff2coins, axis=0)  # min difference of distances
        min_coin, dist2min_coin = np.argmin(np.sum(diff2coins, axis=1)), np.min(np.sum(diff2coins, axis=1))
        
        # find how many walls are in between (finding the patch between the agent and coin)
        patch_coinagent = arena[ min(coins[min_coin][1], d[1]):max(coins[min_coin][1], d[1])+1,  min(coins[min_coin][0], d[0]):max(coins[min_coin][0], d[0])+1] 
        
        # fill features
        f.append(diffmin_xy[0])
        f.append(diffmin_xy[1])
        f.append(dist2min_coin)
        f.append(np.count_nonzero(patch_coinagent))
        f.append(patch_coinagent.shape[0]* patch_coinagent.shape[1] - np.count_nonzero(patch_coinagent))
        feature.append(f)

    return np.asarray(feature)

##################################################################################################################

def setup(self):
    
    # load weights
    try:
        self.weights = np.load('./agent_code/my_agent/models/weights.npy')
        print("weights loaded")
    except:
        self.weights = []
        print("no weights found ---> create new weights")

    # Define Rewards
    self.inm_R = 0
    self.total_R = 0

def act(self):

    self.logger.info('Picking cool action')

    game_state = self.game_state  # isn't it memory waste calling in each feature extraction for coins, self, arena?
    f1 = feat_1(game_state)

    """
    Idea would be compute more features:
    f1 = ... 
    ...
    ...
    f = stack ( all features) 
    """
    # later no necessary
    if self.weights == []:
        self.weights = np.ones((f1.shape[1], 5 ))  # 5 because for the moment only 5 actions

    self.logger.info('Pick action ')
    actions = ['WAIT','RIGHT', 'LEFT', 'DOWN', 'UP']

    self.next_action = np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'], p=[.25, .25, .25, .25])
    
    # load features 
    #coins = self.game_state['coins']
    #x, y, _, bombs_left = self.game_state['self']
    #arena = self.game_state['arena']
    #feat_coins = feature_extraction(coins, arena, directions)
    #directions = [(x,y), (x+1,y), (x-1,y), (x,y+1), (x,y-1)] # define directions
#    best_action = actions[f1.argmin()]
#    self.next_action = best_action
#    self.next_action = np.random.choice([best_action, 'RIGHT', 'LEFT', 'UP', 'DOWN'], p=[.8, .05, .05, .05, .05])

def reward_update(self):

    '''
        IMPORTANT: 
            reward_update happens AFTER act
            This means, self.next_action in reward_update is the
            action the algorithm just took in action. This is why
            we should rename self.next_action as prev_action in 
            reward_update
        
    '''
    reward = 0 
    for event in self.events:
        if event == e.INVALID_ACTION:
            reward -= 10
        elif event == e.COIN_COLLECTED:
            reward += 100
        elif event == e.WAITED:
            reward -= 5 
        else:
            reward -= 1
    
    self.inm_R = reward
    self.total_R += reward


    prev_action = self.next_action
    
    self.w = QLearn_Lin_Approx(self.prev_state, prev_action, self.game_state, reward, w, f)

def end_of_episode(self):
    np.save('./agent_code/my_agent/models/weights.npy', self.weights)
    
