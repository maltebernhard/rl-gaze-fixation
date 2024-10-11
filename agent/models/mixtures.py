import numpy as np
from agent.base_model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class TargetFollowingObstacleEvasionStopMixtureModel(Model):
    id = "TOS"
    observation_keys = ["obstacle1_distance", "robot_target_distance"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.base_env_config["num_obstacles"]
        self.action_space_dimensionality = self.env.action_space.shape[0]

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        evasion_margin_max = 5
        evasion_margin_min = 1
        relevance_obstacle_evasion = min(abs((min(0.0, state[0]-evasion_margin_max)/(evasion_margin_max-evasion_margin_min))**4), 1.0)

        relevance_stop = max(0.0, 1.0 - state[1]/1.0)

        return np.array([max(0.0,1.0-relevance_obstacle_evasion-relevance_stop), max(0.0,relevance_obstacle_evasion-relevance_stop), relevance_stop]), None