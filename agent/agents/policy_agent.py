import gymnasium as gym
import numpy as np
from agent.structure_agent import StructureAgent
import agent.models.policies as policies

# ===================================================================================

class PolicyAgent(StructureAgent):
    models = policies
    
    def __init__(self, base_agent, agent_config) -> None:
        super().__init__(base_agent, agent_config)
        self.action_space_dimensionality = len(self.model_action_keys)
        self.initialize()

    def create_action_space(self):
        return gym.spaces.Box(
            low=np.array([-1.0 for _ in range(self.action_space_dimensionality)]),
            high=np.array([1.0 for _ in range(self.action_space_dimensionality)]),
            dtype=np.float64,
            shape=(self.action_space_dimensionality,)
        )

    def transform_action(self, action, observation) -> np.ndarray:
        return action