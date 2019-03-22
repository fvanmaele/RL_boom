import numpy as np
import pickle
import os

'''
Collects interesting data (game state, action, events) during game runtime 
and saves them as a file per step in a given dirname(will be saved within 
the data-collection folder). This should be called in the callbacks.py in 
the function reward_update of the agent you want to collect data from
'''
def save_game_data(game_state, events, action, dirname):
    
    data = {'state':game_state, 'events':events, 'action':action}
    filename = 'data-collection/game-data/' + dirname + '/data'+str(game_state['step'])
    
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()

'''
Compiles the step-data collected using save_game_data into a one file.
parameters:
    directory: dirname of the step-data
    save_file_name: file name of where all the data will be compiled
'''    
def save_to_file(directory, save_file_name):
    save_file = open(save_file_name, "wb")
    game = []
    
    for filename in os.listdir(directory):
        file = open(os.path.join(directory,filename), 'rb')
        game_data = pickle.load(file)
        game.append(game_data)

    pickle.dump(game, save_file)
    save_file.close()

'''
turn saved game data into a numpy matrix representation
'''

def state2map(data):

    game_state = data['state']
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
    '''
    It turns out that leaving the map as is gives us a flipped,
    rotated map of the area. We can see this if we plot the map
    as an image with plt.imshow. Thus, we need to rotate and flip 
    the map to give us the proper view of the area
    '''
    map = np.rot90(map)
    map = np.flipud(map)
    
    return map
     
    
'''
In an attempt to reduce dimensions of feature vector, this vector representation shows where intereting instances 
(coins, agents, bombs) are in a flattened map of the area
(vector with dimension 289 instead of 17x17 matrix). 
'''     
def state2vec(data):
    vec = []
    self = data['state']['self']
    vec.append(self[0]+self[1]*17)
    
    coins = data['state']['coins']
    for coin in coins:
        vec.append(coin[0]+coin[1]*17) 
    if 9-len(coins)>0:
        vec+= [0]*(9-len(coins))
        
    others = data['state']['others']
    for other in others:
        vec.append(other[0]+other[1]*17)
    if 3-len(others)>0:
        vec+= [0]*(3-len(others))
        
        
    bombs = data['state']['bombs']
    for b in bombs:
        vec.append(b[0]+b[1]*17)
    if 4-len(bombs)>0:
        vec+= [0]*(4-len(bombs))
        
    if len(vec)!=17:
        print(vec, d['state']['bombs'])
    
    return vec