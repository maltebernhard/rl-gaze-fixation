import numpy as np
from agent.models.model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class TowardsTargetModel(Model):
    observation_keys = ["target_offset_angle"]

    def __init__(self, env: StructureEnv, action_space_dimensionality: int = 3):
        super().__init__(env)
        self.action_space_dimensionality = action_space_dimensionality

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        offset_angle = state[0]
        # generate unit length vector in the direction of the target
        if self.action_space_dimensionality == 2:
            action = np.array([np.cos(offset_angle), np.sin(offset_angle)])
        elif self.action_space_dimensionality == 3:
            action = np.array([np.cos(offset_angle), np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None