import numpy as np
from typing import List
import gymnasium as gym
from env.env import Environment

class Agent(gym.Env):
    def __init__(self):
        super().__init__()
        self.env : Environment = gym.make(id='CatcherEnv')
        self.metadata = self.env.metadata
        
        self.total_reward = 0.0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.total_reward += reward
        state = self._get_state(obs)
        return state, reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.total_reward = 0.0
        obs, info = self.env.reset(seed=seed, **kwargs)
        return self._get_state(obs), info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

    def _get_state(self, observation):
        return observation
            
    def env_attr(self, attr):
        return self.env.get_wrapper_attr(attr)