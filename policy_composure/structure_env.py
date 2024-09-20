from abc import abstractmethod
from typing import Dict, List
import gymnasium as gym
import numpy as np

from agent.agent import StructureAgent
from environment.env import Observation

class StructureEnv(gym.Env):
    def __init__(self, env: gym.Env, observation_keys = None):
        super().__init__()
        self.env = env
        self.observations: Dict[str, Observation] = self.env.get_wrapper_attr("observations")
        self.action_space = self.create_action_space()
        self.observation_space = self.create_observation_space(observation_keys)
        self.last_observation = None

    def step(self, action: np.ndarray):
        action = self.transform_action(action)
        # save step returns
        self.last_observation, reward, done, truncated, info = self.env.step(action)
        return self.last_observation[self.observation_indices], reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.last_observation, info = self.env.reset(seed=seed, **kwargs)
        return self.last_observation[self.observation_indices], info
    
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

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

    @abstractmethod
    def create_action_space(self):
        raise NotImplementedError
    
    def create_observation_space(self, observation_keys):
        if observation_keys is None:
            return self.env.observation_space
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
            return gym.spaces.Box(
                low=np.array([obs.low for obs in observation_dict.values()]).flatten(),
                high=np.array([obs.high for obs in observation_dict.values()]).flatten(),
                dtype=np.float64
            )

class MixtureEnv(StructureEnv):
    def __init__(self, env: gym.Env, experts, observation_keys=None, mixture_mode=1):
        self.experts: List[StructureAgent] = experts
        self.mixtures = mixture_mode
        self.mixture_mode = mixture_mode
        if mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif mixture_mode == 2:
            self.action_space_dimensionality = self.env.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        super().__init__(env, observation_keys)
    
    def create_action_space(self):
        return gym.spaces.Box(
            low=np.array([0.0 for _ in range(len(self.experts)*self.action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(len(self.experts*self.action_space_dimensionality))]).flatten(),
            dtype=np.float64
        )
    
    def transform_action(self, action):
        weights = self.normalize_weights(action)
        # get mixture experts' actions
        actions = []
        # n x m array of actions
        for expert in self.experts:
            actions.append(expert.predict(self.last_observation)[0])
        # apply mixture
        env_action = np.sum(weights * actions, axis = 0)
        return env_action
    
    # ============================ helpers =============================

    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = weights.reshape((len(self.experts),1)).repeat(self.env.action_space.shape[0], axis=1)
        else:
            weights = weights.reshape((len(self.experts),self.env.action_space.shape[0]))
        # prevent 0-weights
        sum = np.sum(weights, axis=0)
        weights[:, sum == 0] = 1 / weights.shape[0]
        sum = np.sum(weights, axis=0)
        # normalize weights
        weights = weights / sum
        return weights

