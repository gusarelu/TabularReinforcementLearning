import numpy as np

class Agent(object): #Keep the class name!
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space

    def observe(self, observation, reward, done):
        #Add your code here
        pass
    def act(self, observation):
        #Add your code here

        return np.random.randint(self.action_space)