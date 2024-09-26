import numpy as np
from agent.models.model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class FiftyFiftyMixtureModel(Model):
    observation_keys = []

    def __init__(self, env: StructureEnv, mixture_mode: int = 1, action_space_dimensionality: int = 3):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["num_obstacles"]
        self.mixture_mode = mixture_mode
        self.action_space_dimensionality = action_space_dimensionality

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        if self.mixture_mode == 1:
            weights = np.array([0.5])
        elif self.mixture_mode == 2:
            if self.action_space_dimensionality == 3:
                weights = np.array([0.5, 0.5, 0.5])
            elif self.action_space_dimensionality == 2:
                weights = np.array([0.5, 0.5])
            else:
                raise Exception("Action space dimensionality not supported.")
        else:
            raise Exception("Mixture mode not supported.")
        return weights, None