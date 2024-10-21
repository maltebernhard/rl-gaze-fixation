import types
import gymnasium as gym
import numpy as np
from typing import List
from agent.structure_agent import StructureAgent
import agent.models.mix2res as mixtures
from agent.base_model import Model

# =========================================================================================================

class MixtureOfTwoExperts(StructureAgent):
    def __init__(self, base_agent, agent_config, experts):
        self.experts: List[StructureAgent] = experts
        self.mixture_mode = agent_config["mixture_mode"]
        self.models = mixtures
        super().__init__(base_agent, agent_config)

    def create_action_space(self):
        if self.mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif self.mixture_mode == 2:
            self.action_space_dimensionality = self.experts[0].env.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        return gym.spaces.Box(
            low=np.array([0.0 for _ in range(self.action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(self.action_space_dimensionality)]).flatten(),
            dtype=np.float64
        )

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = np.array([[action[0]], [1.0 - action[0]]])
            weights = weights.reshape((len(self.experts),1)).repeat(self.action_space_dimensionality, axis=1)
        elif self.mixture_mode == 2:
            weights = np.array([action, np.ones(action.shape) - action])
        # get mixture experts' actions
        actions = []
        # n x m array of actions
        for expert in self.experts:
            actions.append(expert.predict_transformed_action(observation)[0])
        # apply mixture
        action = np.sum(weights * actions, axis = 0)
        return action