from abc import abstractmethod
from typing import Dict, List
import gymnasium as gym
import numpy as np
from environment.env import Observation

class StructureEnv(gym.Env):
    def __init__(self, base_env: gym.Env, observation_keys = None):
        super().__init__()
        self.base_env = base_env
        self.observations: Dict[str, Observation] = self.base_env.get_wrapper_attr("observations")
        self.action_space = self.create_action_space()
        self.observation_space = self.create_observation_space(observation_keys)
        self.last_observation = None

    def step(self, action: np.ndarray):
        action = self.transform_action(action)
        # save step returns
        self.last_observation, reward, done, truncated, info = self.next_env.step_full_observation(action)
        return self.last_observation[self.observation_indices], reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.last_observation, info = self.next_env.reset_full_observation(seed=seed, **kwargs)
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

    @abstractmethod
    def create_action_space(self):
        raise NotImplementedError
    
    def create_observation_space(self, observation_keys):
        if observation_keys is None:
            return self.base_env.observation_space
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
        
    def set_next_env(self, next_env):
        self.next_env = next_env
        
# ========================================================================================

class PolicyEnv(StructureEnv):
    def __init__(self, base_env: gym.Env, observation_keys=None):
        super().__init__(base_env, observation_keys)
    
    def transform_action(self, action):
        return action
    
    def create_action_space(self):
        # TODO: make this more general
        return gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64,
            shape=(2,)
        )
    
# ========================================================================================
    
class ContingencyEnv(StructureEnv):
    def __init__(self, base_env: gym.Env, agent, observation_keys=None):
        super().__init__(base_env, observation_keys)
        self.agent = agent
    
    def transform_action(self, action):
        previous_action = self.agent.predict(self.base_env.last_observation)[0]
        return np.concatenate([previous_action, action])
    
    def create_action_space(self):
        # TODO: make this more general
        return gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float64,
            shape=(1,)
        )
    
# ========================================================================================

class MixtureEnv(StructureEnv):
    def __init__(self, base_env: gym.Env, experts, observation_keys: List[str] = None, mixture_mode: int = 1):
        self.experts = experts
        self.mixture_mode = mixture_mode
        if mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif mixture_mode == 2:
            self.action_space_dimensionality = self.base_env.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        super().__init__(base_env, observation_keys)
    
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
            # TODO: make general - currently suboptimal solution
            a = expert.predict(self.base_env.last_observation)[0]
            actions.append(expert.env.transform_action(a))
        # apply mixture
        env_action = np.sum(weights * actions, axis = 0)
        return env_action
    
    # ============================ helpers =============================

    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = weights.reshape((len(self.experts),1)).repeat(self.base_env.action_space.shape[0], axis=1)
        else:
            weights = weights.reshape((len(self.experts),self.base_env.action_space.shape[0]))
        # prevent 0-weights
        sum = np.sum(weights, axis=0)
        weights[:, sum == 0] = 1 / weights.shape[0]
        sum = np.sum(weights, axis=0)
        # normalize weights
        weights = weights / sum
        return weights

