import numpy as np
from settings import s
import copy


##################### HELPING FUNCTIONS

def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.
    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.
    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    USEFUL FOR feature1, feature4
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x,y) for (x,y) in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if free_space[x,y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
        
        
        
def get_blast_coords(bomb, arena, arr):
    x, y = bomb[0], bomb[1]
    if len(arr)== 0:
       arr = np.array([[x,y]])
       #np.append(a, [[0,1]], axis=0)
    
    for i in range(1, 3+1):
        if arena[x+i,y] == -1: break
        arr = np.append(arr,[[x+i,y]], axis=0)
    for i in range(1, 3+1):
        if arena[x-i,y] == -1: break
        arr = np.append(arr,[[x-i,y]], axis=0)            
    for i in range(1, 3+1):
        if arena[x,y+i] == -1: break
        arr = np.append(arr,[[x,y+i]], axis=0)            
    for i in range(1, 3+1):
        if arena[x,y-i] == -1: break
        arr = np.append(arr,[[x,y-i]], axis=0)
    return arr

########################## FEATURE EXTRACTION FUNCTION 

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
            print(d)
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
    
        '''
        TO WRITE
        '''
        print(new_dist_to_safe, dist_to_safe)
        res[np.where(new_dist_to_safe-dist_to_safe<0)] = 1
        
    return res
    

