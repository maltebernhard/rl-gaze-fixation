import numpy as np
import gymnasium as gym
from agent.agents.agent import StructureAgent
from stable_baselines3 import PPO
from agent.models.gaze_fixation import GazeFixationModel

# =============================================================================

class Contingency(StructureAgent):
    def __init__(self, base_agent, agent_config, callback, contingent_agent) -> None:
        self.contingent_agent: StructureAgent = contingent_agent
        super().__init__(base_agent, agent_config, callback)

    def create_action_space(self):
        # TODO: make this more general
        return gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float64,
            shape=(1,)
        )

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        elif self.config["model_type"] == "GFM":
            # TODO: make more general - remove dependency on GazeFixEv
            self.model = GazeFixationModel(self.env, max_vel_rot=self.env.unwrapped.base_agent.env_config["robot_max_vel_rot"], max_acc_rot=self.env.unwrapped.base_agent.env_config["robot_max_acc_rot"])
        else:
            raise ValueError("Model type not supported for Contingency Agent")
        
    def get_observation_keys(self):
        if self.config["model_type"] == "PPO":
            return self.config["observation_keys"]
        elif self.config["model_type"] == "GFM":
            return GazeFixationModel.observation_keys
        else:
            return None

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        previous_action = self.contingent_agent.predict_full_observation(observation)[0]
        action = np.concatenate([self.contingent_agent.transform_action(previous_action, observation), action])
        return action