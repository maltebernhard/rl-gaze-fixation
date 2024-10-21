import numpy as np
from agent.base_model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class TargetFollowingObstacleEvasionMixtureModel(Model):
    id = "TOM"
    observation_keys = [item for sublist in [[f"obstacle{i+1}_offset_angle", f"obstacle{i+1}_distance"] for i in range(100)] for item in sublist]
    action_keys = ["relevance_obstacle_evasion"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.base_env_config["num_obstacles"]
        self.action_space_dimensionality = self.env.action_space.shape[0]

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        nearest_obstacle = 0
        for i in range(1,self.num_obstacles):
            if state[2*i+1] > state[2*nearest_obstacle+1]:
                nearest_obstacle = i
        obstacle_coverage = state[2*nearest_obstacle+1]
        relevance_obstacle_evasion = min(obstacle_coverage**2 * 12, 1.0)

        if self.action_space_dimensionality == 1:
            weights = np.array([relevance_obstacle_evasion])
        elif self.action_space_dimensionality == 2:
            weights = np.array([relevance_obstacle_evasion, relevance_obstacle_evasion])
        elif self.action_space_dimensionality == 3:
            weights = np.array([relevance_obstacle_evasion, relevance_obstacle_evasion, 0.5])
        else:
            raise Exception("Action space dimensionality not supported.")
        return weights, None
    
# =============================================================================

class FiftyFiftyMixtureModel(Model):
    id = "55M"
    observation_keys = []
    action_keys = []

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.action_space_dimensionality = self.env.action_space.shape

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        weights = 0.5 * np.ones(self.action_space_dimensionality)
        return weights, None