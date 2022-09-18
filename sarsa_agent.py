import random
import numpy as np
from contextlib import contextmanager


class Agent(object):  # Keep the class name!
    """SARSA agent"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.Q_table = np.zeros((state_space, action_space))
        if state_space == 6:
            self.Q_table += 10.0  # optimistic initialization only for riverswim
        self.alpha = 0.1
        self.gamma = .99
        self.epsilon = 0.05
        self.action = None

    def observe(self, observation, reward, done):
        next_action = self.choose_next_action(observation)
        update = reward + self.gamma * self.Q_table[observation, next_action] * (not done)
        self.Q_table[self.state][self.action] += self.alpha * (update - self.Q_table[self.state][self.action])
        self.action = None if done else next_action  # if episode is done reset action to None

    def act(self, observation):
        self.state = observation
        if self.action is None:  # start of episode
            self.action = self.choose_next_action(observation)
        elif self.epsilon == 0:  # meaning not training (no call to self.observe())
            self.action = self.choose_next_action(observation)
        return self.action

    def choose_next_action(self, observation):
        if random.random() < self.epsilon:
            # Exploration
            action = random.randrange(self.action_space)
        else:
            # Exploitation
            q = self.Q_table[observation]
            candidates = np.nonzero(q == np.max(q))[0]
            action = random.choice(candidates)  # arbitrary tie-breaker
        return action

    def save_Q_table(self, path):
        np.save(path, self.Q_table)

    @contextmanager
    def greedy(self):
        # temporarily setting epsilon to zero
        self.epsilon, epsilon_ = 0, self.epsilon
        yield
        # returning epsilon to its original value
        self.epsilon = epsilon_
