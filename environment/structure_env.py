from abc import abstractmethod
from typing import Dict, List, Tuple
import gymnasium as gym
import numpy as np
from environment.gaze_fix_env import Observation

class StructureEnv(gym.Env):
    def __init__(self, base_env: gym.Env, observation_keys = None, action_space = None):
        super().__init__()
        self.base_env = base_env
        self.create_observation_space(observation_keys)
        self.action_space = action_space
        self.last_observation = None
        self.last_action: np.ndarray = None
        self.current_action = None
    
    def step(self, action: np.ndarray):
        self.last_action = action.copy()
        self.last_observation, reward, done, truncated, info = self.base_env.step(action)
        return self.last_observation[self.observation_indices], reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.last_action = None
        self.last_observation, info = self.base_env.reset(seed=seed, **kwargs)
        return self.last_observation[self.observation_indices], info
    
    def render(self):
        self.base_env.render()

    def close(self):
        self.base_env.close()

    # ======================================================================================
    
    @abstractmethod
    def transform_action(self, action):
        raise NotImplementedError
    
    def create_observation_space(self, observation_keys):
        if observation_keys is None:
            # use all available observations
            self.observation_space = self.base_env.observation_space
            self.observation_indices = np.array([i for i in range(len(self.base_env.get_wrapper_attr("observations").keys()))])
        else:
            observation_dict: Dict[str, Observation] = {}
            all_observations: Dict[str, Observation] = self.base_env.get_wrapper_attr("observations")
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