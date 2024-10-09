import gymnasium as gym
import numpy as np
from typing import List
from agent.agents.agent import StructureAgent
from stable_baselines3 import PPO

# =========================================================================================================

class MixtureOfExperts(StructureAgent):
    def __init__(self, base_agent, agent_config, callback, experts):
        self.experts: List[StructureAgent] = experts
        self.mixture_mode = agent_config["mixture_mode"]
        super().__init__(base_agent, agent_config, callback)

    def create_action_space(self):
        if self.mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif self.mixture_mode == 2:
            self.action_space_dimensionality = self.env.unwrapped.base_env.unwrapped.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        return gym.spaces.Box(
            low=np.array([0.0 for _ in range(len(self.experts)*self.action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(len(self.experts)*self.action_space_dimensionality)]).flatten(),
            dtype=np.float64
        )

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        else:
            raise ValueError("Model type not supported for Mixture-of-Experts Agent")
        
    def get_observation_keys(self):
        if self.config["model_type"] == "PPO":
            return self.config["observation_keys"]
        else:
            return None

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        weights = self.normalize_weights(action)
        # get mixture experts' actions
        actions = []
        # n x m array of actions
        for expert in self.experts:
            a = expert.predict_full_observation(observation)[0]
            actions.append(expert.transform_action(a, observation))
        # apply mixture
        action = np.sum(weights * actions, axis = 0)
        return action
    
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = weights.reshape((len(self.experts),1)).repeat(self.experts[0].env.action_space.shape[0], axis=1)
        else:
            weights = weights.reshape((len(self.experts),self.experts[0].env.action_space.shape[0]))
        # prevent 0-weights
        sum = np.sum(weights, axis=0)
        weights[:, sum == 0] = 1 / weights.shape[0]
        sum = np.sum(weights, axis=0)
        # normalize weights
        weights = weights / sum
        return weights