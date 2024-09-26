from abc import abstractmethod
from datetime import datetime
from typing import List
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from agent.models.model import Model
from utils.callback import PlottingCallback, ModularAgentCallback
from environment.structure_env import StructureEnv

# ===========================================================================================

class StructureAgent:
    def __init__(self, base_env, agent_config: dict, callback: ModularAgentCallback) -> None:
        self.id = agent_config["id"]
        self.config = agent_config
        self.observation_keys = None
        self.env: StructureEnv = gym.make(
            id='StructureEnv',
            base_env = base_env,
            observation_keys = self.get_observation_keys(),
            action_space = self.create_action_space(),
            reward_indices = agent_config["reward_indices"] if "reward_indices" in agent_config else np.array([0,1,2])
        )
        self.observation_indices = self.env.get_wrapper_attr("observation_indices")
        self.model: Model = None
        self.set_model()
        self.model_name = agent_config["model_type"]
        self.set_callback(callback)

    def run(self, prints = False, steps = 0, env_seed = None):
        total_reward = 0
        step = 0
        obs, info = self.env.reset(seed=env_seed)
        done = False
        while not done and (steps==0 or step < steps):
            action, _states = self.predict(obs)
            if prints:
                print(f'-------------------- Step {step} ----------------------')
                print(f'Observation: {obs}')
                print(f'Action:      {action}')
            obs, reward, done, truncated, info = self.env.step(action)
            if prints:
                print(f'Reward:      {reward}')
            total_reward += reward
            step += 1
            self.env.render()
            
        #obs, info = self.env.reset_full_observation()
        self.env.close()
        obs, info = self.env.reset()
        print(f"Episode finished with total reward {total_reward}")

    def learn(self, total_timesteps) -> None:
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

    def save(self, folder = None):
        if folder is None:
            folder = "./training_data/" + datetime.today().strftime('%Y-%m-%d_%H-%M') + "/"
        filename = self.model_name
        self.model.save(folder + filename)

    def load(self, filename):
        if self.model_name == "PPO":
            self.model = PPO.load(filename, env=self.env)

    def predict(self, observation: np.ndarray, deterministic = True) -> np.ndarray:
        action = self.model.predict(observation, deterministic)
        return action
    
    def predict_full_observation(self, observation: np.ndarray) -> np.ndarray:
        # if last_action is set, this model is actively controlling the environment
        if self.env.unwrapped.last_action is not None:
            action = self.env.unwrapped.last_action.copy(), None
            self.env.unwrapped.last_action = None
        # if it isn't, predict action again
        else:
            observation = observation[self.observation_indices]
            action = self.model.predict(observation)
        return action

    def set_callback(self, callback=None) -> None:
        if callback is None:
            self.callback = PlottingCallback(self.model_name)
        else:
            self.callback = callback

    @abstractmethod
    def get_observation_keys(self) -> List[str]:
        raise NotImplementedError
    
    @abstractmethod
    def create_action_space(self, mixture_mode, num_experts):
        raise NotImplementedError
    
    @abstractmethod
    def set_model(self):
        raise NotImplementedError

    @abstractmethod
    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError
