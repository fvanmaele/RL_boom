import pickle
import os
import numpy as np


'''
We decided to save the policies we've learned in a class structure.
    A policy has:
        a game state just as the map of the area (matrix representation using numpy),
        action that should be taken in such state (string),
        reward in this situation for training(double)
'''
class Policy:

    def __init__(self, state, action, reward=0.0):
        self.state = state
        self.action = action
        self.reward = reward
    
'''
turn saved game data into 
'''
def game_data_to_policy(filename):

    file = open(filename, "rb")
    data = pickle.load(file)
    file.close()

    game_state = data['state']
    action = data['action']
    '''
    create a map of the area
    '''
    map = game_state['arena']
    '''
    write where explosions will occure
    '''
    explosions = game_state['explosions']
    map[np.argwhere(explosions>0)] = explosions[np.argwhere(explosions>0)]


    current_player = game_state['self']
    '''
    save the player locations.
        current player: 10
        other players : 11, 12, 13 
    '''
    map[current_player[0], current_player[1]] = 10

    '''
    check if there are any other players, 
    if there are, save their locations in map as described above
    '''
    others = game_state['others']
    if len(others)>0:
        for i in range(len(others)):
            other_player = others[i]
            map[other_player[0], other_player[1]] = 10+i+1

    '''
    save bomb locations as ticker+20
    '''
    bombs = game_state['bombs']
    if len(bombs)>0:
        for bomb in bombs:
            map[bomb[0], bomb[1]] = bomb[2]+20
            
    '''
    save coin locations as 30
    '''
    coins = game_state['coins']
    if len(coins)>0:
        for coin in coins:
            map[coin[0], coin[1]] = 30
            
            
    new_policy =  Policy(map, action)
    return new_policy
    
def save_to_file(directory, filename):
    save_file = open(filename, "wb")
    Policies = []
    for filename in os.listdir(directory):
        file = os.path.join(directory,filename)
        new_policy = game_data_to_policy(file)
        Policies.append(new_policy)

    pickle.dump(Policies, save_file)
    save_file.close()
    
    