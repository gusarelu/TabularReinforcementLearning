import numpy as np
import random
from contextlib import contextmanager


class Agent(object):  # Keep the class name!
    """Epsilon-greedy agent with updating exploration rate"""

    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.Q_table = np.zeros((state_space, action_space))
        self.alpha = 0.1
        self.gamma = .99
        self.eps = 1
        self.eps_min = 0.01
        self.eps_decay = 0.001

    def observe(self, observation, reward, done):
        update = reward + self.gamma * np.max(self.Q_table[observation]) * (not done)
        self.Q_table[self.state][self.action] += self.alpha * (update - self.Q_table[self.state][self.action])
        if done:
            self.update_epsilon()


    def act(self, observation):
        if random.random() < self.eps:
            # Exploration
            action = random.randrange(self.action_space)
        else:
            # Exploitation
            q = self.Q_table[observation]
            max_actions = np.nonzero(q == np.max(q))[0]
            action = random.choice(max_actions)  # arbitrary tie breaker
        self.state = observation
        self.action = action
        return action

    def update_epsilon(self):
        self.eps = max(self.eps_min, self.eps - self.eps_decay)


    def save_Q_table(self, path):
        np.save(path, self.Q_table)

    @contextmanager
    def greedy(self):
        eps_ = 0
        self.eps, eps_ = eps_, self.eps
        yield
        self.eps, eps_ = eps_, self.eps


if __name__ == "__main__":
    pass
