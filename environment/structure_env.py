from abc import abstractmethod
from typing import Dict
import gymnasium as gym
import numpy as np
from environment.base_env import Observation

class StructureEnv(gym.Env):
    def __init__(self, base_agent, observation_keys = None, action_space = None, reward_indices = np.array([0,1,2])):
        super().__init__()
        #self.base_env = base_env
        self.base_agent = base_agent
        self.create_observation_space(observation_keys)
        self.action_space = action_space
        self.reward_indices = reward_indices
        self.last_observation = None
        self.last_action: np.ndarray = None
        self.current_action = None
    
    def step(self, action: np.ndarray):
        self.last_action = action.copy()
        self.last_observation, rewards, done, truncated, info = self.base_agent.act()
        return self.last_observation[self.observation_indices], np.sum(rewards[self.reward_indices]), done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.last_action = None
        self.last_observation, info = self.base_agent.reset(seed=seed, **kwargs)
        return self.last_observation[self.observation_indices], info
    
    def render(self):
        self.base_agent.render()

    def close(self):
        self.base_agent.close()

    # ======================================================================================
    
    def set_base_agent(self, agent):
        self.base_agent = agent

    @abstractmethod
    def transform_action(self, action):
        raise NotImplementedError
    
    def create_observation_space(self, observation_keys):
        if observation_keys is None:
            # use all available observations
            self.observation_space = self.base_agent.observation_space
            self.observation_indices = np.array([i for i in range(len(self.base_agent.observations.keys()))])
        else:
            observation_dict: Dict[str, Observation] = {}
            all_observations: Dict[str, Observation] = self.base_agent.observations
            observation_indices = []
            for obskey in observation_keys:
                index = 0
                for key, obs in all_observations.items():
                    if obskey == key:
                        observation_dict[key] = obs
                        observation_indices.append(index)
                        continue
                    index += 1
            self.observation_indices = np.array(observation_indices)
            self.observation_space = gym.spaces.Box(
                low=np.array([obs.low for obs in observation_dict.values()]).flatten(),
                high=np.array([obs.high for obs in observation_dict.values()]).flatten(),
                dtype=np.float64
            )