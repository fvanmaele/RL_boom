




def feature_4(state, action):
    """
    This feature rewards the action that minimizes the distance to safety
    should the agent be in the danger zone where explosions will be).
        F(s,a) = 1, should a reduces distance to safety
        F(s,a) = 0, otherwise
    F(s,a) returns 0 if we are not in the danger zone   
    """
    agent = game_state['self']
    arena = game_state['arena']
    bombs = game_state['bombs']

    # Initialize result vector
    res = np.zeros(6, dtype=np.int8)

    if len(bombs) != 0:
        """
        If there are bombs on the game board, we map all the explosions
        that will be caused by the bombs. For this, we use the help function
        get_blast_coords. This is an adjusted version of the function with the
        same name from item.py of the original framework
        """
        danger_zone = []
        for b in bombs:
            danger_zone = get_blast_coords(b, arena, danger_zone)

        """
        We then check if the agent is in the danger zone
        If agent is not in the danger zone, we return 0.
        Otherwise we calculate the distance/direction of safety.
        """
        if agent[0] in danger_zone[:,0] and agent[1] in danger_zone[:,1]:            
            """
            We then mark these explosions on our map. here we deep-copy
            the arena, so that in the case that the arena is needed for
            other features, it remains unchanged
            """
            map_ = copy.deepcopy(arena)
            map_[danger_zone[:,0], danger_zone[:,1]] = 2

            safe_loc = np.argwhere(map_==0)
            free_space = abs(map_) != 1
            d = look_for_targets(free_space, (agent[0], agent[1]), safe_loc)
            print(d)

            """
            We then calculate the minimum distance of our agent to any
            of these safe locations.  For simplicity, only Manhattan
            distance is used to calculate distance in this feature
            extraction. However, this doesn't always represent the
            true distance in the game because of the walls.(Possible
            point of improvement?)
 
            Then, we calculate the positions of our agent after taking
            all possible actions.
            """
            
            actions_loc = np.array([(agent[0], agent[1]-1), #up
                                    (agent[0], agent[1]+1), #down
                                    (agent[0]-1, agent[1]), #left
                                    (agent[0]+1, agent[1]), #right
                                    (agent[0], agent[1]),   #bomb
                                    (agent[0], agent[1])])  #wait
            
            res = (actions_loc[:,0] == d[0]) & (actions_loc[:,1] == d[1])
            res = res.astype(int)
            
        return res





def feature_9(state, action):
    """Feature 0

    Return Value:
    * 1:  Action is most likely to kill another agent.
    * 0:  Otherwise.

    Compared to crates, the difficulty here is that agents can
    move and drop bombs.
    """
    pass


def feature_10(state, action):
    """Feature 10

    Value: The distance between our agent's position and
           one of the other agents.
    """
    pass

def feature_11(state, action):
    pass

def feature_12(state, action):
    pass

def feature_13(state, action):
    pass

# TODO: implement or not?
def feature_14(state, action):
    """Feature 14

    Value: The time remaining to the end of episode.
    """
    pass
