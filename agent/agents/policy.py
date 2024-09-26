import gymnasium as gym
import numpy as np
from agent.agents.agent import StructureAgent
from stable_baselines3 import PPO
from agent.models.avoid_nearest_obstacle import AvoidNearestObstacleModel
from agent.models.towards_target import TowardsTargetModel

# ===================================================================================

class Policy(StructureAgent):
    def __init__(self, base_env, agent_config, callback) -> None:
        super().__init__(base_env, agent_config, callback)

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

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        # TODO: fix hard-coded action space dimensionality and number of obstacles
        elif self.config["model_type"] == "ANO":
            self.model = AvoidNearestObstacleModel(self.env, action_space_dimensionality=self.config["action_space_dimensionality"])
        elif self.config["model_type"] == "GTT":
            self.model = TowardsTargetModel(self.env, action_space_dimensionality=self.config["action_space_dimensionality"])
        else:
            raise ValueError("Model type not supported for Policy Agent")

    def get_observation_keys(self):
        if self.config["model_type"] == "PPO":
            return self.config["observation_keys"]
        elif self.config["model_type"] == "ANO":
            return AvoidNearestObstacleModel.observation_keys
        elif self.config["model_type"] == "GTT":
            return TowardsTargetModel.observation_keys
        else:
            return None

    def transform_action(self, action, observation) -> np.ndarray:
        return action