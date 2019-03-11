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
    
