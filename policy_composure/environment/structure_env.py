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
        self.observation_space = self.create_observation_space(observation_keys)
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

# ========================================================================================

class PolicyEnv(StructureEnv):
    def __init__(self, base_env: gym.Env, observation_keys=None):
        super().__init__(base_env, observation_keys)
        self.create_action_space()
    
    def create_action_space(self):
        # TODO: make this more general
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64,
            shape=(2,)
        )
    
# ========================================================================================
    
class ContingencyEnv(StructureEnv):
    def __init__(self, base_env: gym.Env, contingent_agent, observation_keys=None):
        super().__init__(base_env, observation_keys)
        #self.contingent_agent = contingent_agent
        self.create_action_space()
    
    def create_action_space(self):
        # TODO: make this more general
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float64,
            shape=(1,)
        )
    
# ========================================================================================

class MixtureEnv(StructureEnv):
    def __init__(self, base_env: gym.Env, experts, observation_keys: List[str] = None, mixture_mode: int = 1):
        # self.experts = experts
        # self.mixture_mode = mixture_mode
        super().__init__(base_env, observation_keys)
        self.create_action_space(mixture_mode, experts)
    
    def create_action_space(self, mixture_mode, experts):
        if mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif mixture_mode == 2:
            self.action_space_dimensionality = self.base_env.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        self.action_space = gym.spaces.Box(
            low=np.array([0.0 for _ in range(len(experts)*self.action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(len(experts*self.action_space_dimensionality))]).flatten(),
            dtype=np.float64
        )
        