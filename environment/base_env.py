import gymnasium as gym
import numpy as np

class BaseEnv(gym.Env):
    def __init__(self, env: gym.Env):
        self.env = env
        self.base_agent = None
        self.observations = self.env.get_wrapper_attr("observations")
        self.config = self.env.get_wrapper_attr("config")

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.last_observation = None
    
    # TODO: find better way to ignore this partial action
    def step(self, partial_action):
        action = self.base_agent.predict(self.last_observation)[0]
        self.last_observation, rewards, done, truncated, info = self.env.step(action)
        return self.last_observation, rewards, done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.last_observation, info = self.env.reset(seed=seed, **kwargs)
        return self.last_observation, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    
    def set_base_agent(self, agent):
        self.base_agent = agent