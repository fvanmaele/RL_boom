import numpy as np
from random import shuffle
from settings import e
from settings import s
from agent_code.my_agent.algorithms import *
from agent_code.my_agent.feature4 import *


def feature2(game_state, bomb_map):
    x, y, _, bombs_left = game_state['self']
    directions = [(x,y-1), (x,y+1), (x-1,y), (x+1,y), (x,y)]
    bombs = game_state['bombs']
    others = [(x,y) for (x,y,n,b) in game_state['others']]
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    explosions = game_state['explosions'] 
    arena = game_state['arena']
    #explosion_map = game_state[

    feature = []

    for d in directions:
        # if invalid action agent should wait
        if ((arena[d] == 0) and
            (not d in others) and
            (not d in bomb_xys)):
            d = (x,y)

        if ((bomb_map[d] <= 1) and
            (explosions[d] <= 1)):
            feature.append(1) 
        else:
            feature.append(0)
    
    # For BOMB 
    feature.append(feature[-1])

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
    self.total_R = 0

    # Step size or gradient descent 
    self.alpha = 0.2 
    self.gamma = 0.9
    self.EPSILON = 0.2


#    # While this timer is positive, agent will not hunt/attack opponents
#    self.ignore_others_timer = 0

def act(self):

    """
    actions order: 'UP', 'DOWN', LEFT', 'RIGHT', 'BOMB', 'WAIT'    
    """

    # load state 
    game_state = self.game_state  # isn't it memory waste calling in each feature extraction for coins, self, arena?
    
    # create BOMB-MAP 
    bombs = game_state['bombs']
    arena = game_state['arena']
    bomb_xys = [(x,y) for (x,y,t) in bombs]
    bomb_map = np.ones(arena.shape) * 5
    for xb,yb,t in bombs:
        for (i,j) in [(xb+h, yb) for h in range(-3,4)] + [(xb, yb+h) for h in range(-3,4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i,j] = min(bomb_map[i,j], t)
    
    # Compute features state 
    f0 = np.ones(6)  # for bias
    f1 = feature1(game_state) # reward good action
    f2 = feature2(game_state, bomb_map) # penalization bad action
    f4 = feature4(game_state) #reward going towards safe locations
    f7 = feature7(game_state) # penalize bad action    !!!! NOCH MAL SCHAUEN
    f8 = feature8(game_state) # rewards good action
    feature_state = np.vstack((f0,f1,f2,f7,f8)).T
    self.prev_state = feature_state
    print(self.game_state['step']) 
    print(f2)
    
    
    #weights = np.array([1,1,-1,-1,1])   #initial guess 
    # later no necessary
    if len(self.weights) == 0:
        print(feature_state.shape[0])
        weights = np.ones(feature_state.shape[1])  
        self.weights = weights
    else:
        weights = self.weights

    self.logger.info('Pick action')
    
    # Linear approximation approach
    greedy = np.random.choice([0,1], p=[self.EPSILON, 1-self.EPSILON)
    if greedy:
    
        q_approx = linapprox_q(feature_state, weights)
        best_actions = np.where(q_approx == np.max(q_approx))[0] 
        shuffle(best_actions)
        q_next_action = s.actions[best_actions[0]] #GREEDY POLICY
        self.next_action = q_next_action
        print(" q action picked  ", q_next_action)


    else:
        random_action = np.random.choice(['WAIT','RIGHT', 'LEFT', 'DOWN', 'UP','BOMB'], p=[0.15, .15, 0.15, .15, 0.15, 0.25])
        self.next_action = random_action
        print(" random action picked  ", random_action)
    
def reward_update(self):

    '''
        IMPORTANT: 
            reward_update happens AFTER act
            This means, self.next_action in reward_update is the
            action the algorithm just took in action. This is why
            we should rename self.next_action as prev_action in 
            reward_update
        
    '''
    self.logger.info('IN TRAINING MODE ')

    reward = 0 

    if self.game_state['step'] != 1:
        for event in self.events:
            if event == e.INVALID_ACTION:
                reward -= 10 
            elif event == e.COIN_COLLECTED:
                reward += 100
            elif event == e.WAITED:
                reward -= 10 
            else:
                reward -= 1
        
        self.total_R += reward
       
        # load weights
        weights = self.weights

        # learning 

        # Berechene alle features
        next_state = feat_1(self.game_state)
        """
        so wie vorhin und mit stack, vlt, machen wir eine Function die alles auf einmal stack
        """
        prev_state = self.prev_state
        prev_state_a = prev_state[s.actions.index(self.next_action),:]

        alpha = self.alpha
        gamma = self.gamma
        # update weights
        weights = q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma)        

        self.weights = weights
        self.alpha = 1/self.game_state['step']
        self.gamma = self.gamma ** self.game_state['step']
        

def end_of_episode(self):
    np.save('./agent_code/my_agent/models/weights.npy', self.weights)
    

def linapprox_q(state, weights):
    q_approx = np.dot(state, weights)
    return q_approx

def q_gd_linapprox(next_state, prev_state_a, reward, weights, alpha, gamma):
    next_state_a = next_state[np.argmax(linapprox_q(next_state, weights)), :]
    weights += alpha * (reward + gamma * np.dot(next_state_a,weights) - np.dot(prev_state_a,weights)) * prev_state_a 
    return weights
            
#def eps_greedy(self, epsilon = ):
#    return EPSILON/(C*self.round/s.n_rounds+1)
