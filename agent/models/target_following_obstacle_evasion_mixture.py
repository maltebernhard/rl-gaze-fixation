import numpy as np
from agent.models.model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class TargetFollowingObstacleEvasionMixtureModel(Model):
    observation_keys = [item for sublist in [[f"obstacle{i+1}_offset_angle", f"obstacle{i+1}_coverage"] for i in range(100)] for item in sublist]

    def __init__(self, env: StructureEnv, mixture_mode: int = 1, action_space_dimensionality: int = 3):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["num_obstacles"]
        self.mixture_mode = mixture_mode
        self.action_space_dimensionality = action_space_dimensionality

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        nearest_obstacle = 0
        for i in range(1,self.num_obstacles):
            if state[2*i+1] > state[2*nearest_obstacle+1]:
                nearest_obstacle = i
        obstacle_coverage = state[2*nearest_obstacle+1]
        relevance_obstacle_evasion = min(obstacle_coverage**2 * 12, 1.0)

        if self.mixture_mode == 1:
            weights = np.array([relevance_obstacle_evasion])
        elif self.mixture_mode == 2:
            if self.action_space_dimensionality == 3:
                weights = np.array([relevance_obstacle_evasion, relevance_obstacle_evasion, 0.5])
            elif self.action_space_dimensionality == 2:
                weights = np.array([relevance_obstacle_evasion, relevance_obstacle_evasion])
            else:
                raise Exception("Action space dimensionality not supported.")
        else:
            raise Exception("Mixture mode not supported.")
        return weights, None