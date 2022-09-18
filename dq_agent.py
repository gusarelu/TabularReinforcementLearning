import random
import numpy as np
from contextlib import contextmanager


class Agent(object):  # Keep the class name!
    """Double Q-learning agent"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.Q_table1 = np.zeros((state_space, action_space))
        self.Q_table2 = np.zeros((state_space, action_space))
        if state_space == 6:
            self.Q_table1 += 10.0  # optimistic initialization only for riverswim
            self.Q_table2 += 10.0
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.05

    def observe(self, observation, reward, done):
        if random.random() < 0.5:
            Q1 = self.Q_table1
            Q2 = self.Q_table2
        else:
            Q1 = self.Q_table2
            Q2 = self.Q_table1
        next_action = np.argmax(Q1[observation])
        update = reward + self.gamma * Q2[observation, next_action] * (not done)
        Q1[self.state][self.action] += self.alpha * (update - Q1[self.state][self.action])

    def act(self, observation):
        if random.random() < self.epsilon:
            # Exploration
            action = random.randrange(self.action_space)
        else:
            # Exploitation
            q = self.Q_table1[observation] + self.Q_table2[observation]
            candidates = np.nonzero(q == np.max(q))[0]
            action = random.choice(candidates)  # arbitrary tie-breaker
        self.state = observation
        self.action = action
        return action

    def save_Q_table(self, path):
        np.save(path, (self.Q_table1 + self.Q_table2) / 2)

    @contextmanager
    def greedy(self):
        # temporarily setting epsilon to zero
        self.epsilon, epsilon_ = 0, self.epsilon
        yield
        # returning epsilon to its original value
        self.epsilon = epsilon_
