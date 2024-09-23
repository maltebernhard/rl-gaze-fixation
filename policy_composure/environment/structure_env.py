from abc import abstractmethod
from typing import Dict, List
import gymnasium as gym
import numpy as np
from environment.env import Observation

class StructureEnv(gym.Env):
    def __init__(self, base_env: gym.Env, observation_keys = None, action_space = None):
        super().__init__()
        self.base_env = base_env
        self.observations: Dict[str, Observation] = self.base_env.get_wrapper_attr("observations")
        self.create_observation_space(observation_keys)
        self.action_space = action_space
        self.last_observation = None
        self.current_action = None
    
    def step(self, action: np.ndarray):
        self.last_observation, reward, done, truncated, info = self.base_env.step(action)
        return self.last_observation[self.observation_indices], reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.last_observation, info = self.base_env.reset(seed=seed, **kwargs)
        return self.last_observation[self.observation_indices], info
    
    def render(self):
        self.base_env.render()

    def close(self):
        self.base_env.close()

    # ======================================================================================

    def step_full_observation(self, action: np.ndarray):
        _, reward, done, truncated, info = self.step(action)
        return self.last_observation, reward, done, truncated, info
    
    def reset_full_observation(self, seed=None, **kwargs):
        _, info = self.reset(seed=seed, **kwargs)
        return self.last_observation, info
    
    @abstractmethod
    def transform_action(self, action):
        raise NotImplementedError
    
    def create_observation_space(self, observation_keys):
        if observation_keys is None:
            self.observation_space = self.base_env.observation_space
        else:
            index = 0
            observation_dict: Dict[str, Observation] = {}
            observation_indices = []
            for key, obs in self.observations.items():
                if key in observation_keys:
                    observation_dict[key] = obs
                    observation_indices.append(index)
                index += 1
            self.observation_indices = np.array(observation_indices)
            self.observation_space = gym.spaces.Box(
                low=np.array([obs.low for obs in observation_dict.values()]).flatten(),
                high=np.array([obs.high for obs in observation_dict.values()]).flatten(),
                dtype=np.float64
            )