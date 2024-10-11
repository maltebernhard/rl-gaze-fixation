import types
import numpy as np
import gymnasium as gym
from agent.structure_agent import StructureAgent
import agent.models.contingencies as contingencies
from agent.base_model import Model

# =============================================================================

class Contingency(StructureAgent):
    def __init__(self, base_agent, agent_config, callback, contingent_agent) -> None:
        self.contingent_agent: StructureAgent = contingent_agent
        self.models = contingencies
        super().__init__(base_agent, agent_config, callback)

    def create_action_space(self):
        # TODO: make this more general
        return gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float64,
            shape=(1,)
        )

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        previous_action = self.contingent_agent.predict_full_observation(observation)[0]
        action = np.concatenate([self.contingent_agent.transform_action(previous_action, observation), action])
        return action