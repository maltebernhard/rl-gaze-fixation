import types
import gymnasium as gym
import numpy as np
from agent.structure_agent import StructureAgent
from agent.base_model import Model
import agent.models.policies as policies

# ===================================================================================

class Policy(StructureAgent):
    def __init__(self, base_agent, agent_config) -> None:
        self.models = policies
        super().__init__(base_agent, agent_config)

    def create_action_space(self):
        if "action_space_dimensionality" in self.config.keys():
            self.action_space_dimensionality = self.config["action_space_dimensionality"]
        # TODO: remove and include action space dimensionality in all configs
        # what about low and high?
        else:
            self.action_space_dimensionality = 2
        return gym.spaces.Box(
            low=np.array([-1.0 for _ in range(self.action_space_dimensionality)]),
            high=np.array([1.0 for _ in range(self.action_space_dimensionality)]),
            dtype=np.float64,
            shape=(self.action_space_dimensionality,)
        )

    def transform_action(self, action, observation) -> np.ndarray:
        return action