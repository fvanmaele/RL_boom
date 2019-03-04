class Policy:

    def __init__(self, state, action, reward):
        valid_actions = []
        self.state = state
        if action in valid_actions:
            self.action = action
        else:
            print("ERROR: INVALID ACTION")
        self.reward = reward
    
    