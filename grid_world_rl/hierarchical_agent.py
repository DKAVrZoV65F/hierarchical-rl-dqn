# grid_world_rl/hierarchical_agent.py

from .agent import DQNAgent
from .config import (
    HIGH_LEVEL_STATE_SIZE,
    HIGH_LEVEL_ACTION_SIZE,
    LOW_LEVEL_STATE_SIZE,
    LOW_LEVEL_ACTION_SIZE
)

class HighLevelAgent:
    def __init__(self):
        self.agent = DQNAgent(HIGH_LEVEL_STATE_SIZE, HIGH_LEVEL_ACTION_SIZE)

    def choose_option(self, state):
        return self.agent.act(state)

    def remember(self, state, option, reward, next_state, done):
        self.agent.remember(state, option, reward, next_state, done)

    def replay(self, batch_size):
        self.agent.replay(batch_size)

    def update_epsilon(self):
        self.agent.update_epsilon()

    def save(self, name):
        self.agent.save(name)

    def load(self, name):
        self.agent.load(name)

class LowLevelAgent:
    def __init__(self):
        self.agent = DQNAgent(LOW_LEVEL_STATE_SIZE, LOW_LEVEL_ACTION_SIZE)

    def choose_action(self, state):
        return self.agent.act(state)

    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(state, action, reward, next_state, done)

    def replay(self, batch_size):
        self.agent.replay(batch_size)

    def update_epsilon(self):
        self.agent.update_epsilon()

    def save(self, name):
        self.agent.save(name)

    def load(self, name):
        self.agent.load(name)