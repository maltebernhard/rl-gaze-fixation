from abc import abstractmethod
from datetime import datetime
import types
from typing import List, Tuple, Type
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from agent.base_model import Model
from agent.models import policies, contingencies, mixtures, mix2res
from utils.callback import ModularAgentCallback
from environment.structure_env import StructureEnv

# ===========================================================================================

class StructureAgent:
    def __init__(self, base_agent, agent_config: dict, callback: ModularAgentCallback) -> None:
        self.id = agent_config["id"]
        self.config = agent_config
        self.env: StructureEnv = gym.make(
            id='StructureEnv',
            base_agent = base_agent,
            action_space = self.create_action_space(),
            reward_indices = agent_config["reward_indices"] if "reward_indices" in agent_config else np.array([0])
        )
        self.set_model()
        self.set_callback(callback)

    def run(self, timesteps = 0, env_seed = None, render = True, prints = False):
        log = {"actions": [], "observations": [], "rewards": []}
        total_reward = 0
        step = 0
        obs, info = self.env.reset(seed=env_seed)
        done = False
        while not done and (timesteps==0 or step < timesteps):
            action, _states = self.predict(obs)
            log["actions"].append(action)
            log["observations"].append(obs)
            if prints:
                print(f'-------------------- Step {step} ----------------------')
                print(f'Observation: {obs}')
                print(f'Action:      {action}')
            obs, reward, done, truncated, info = self.env.step(action)
            log["rewards"].append(reward)
            if prints:
                print(f'Reward:      {reward}')
            total_reward += reward
            step += 1
            if render:
                self.env.render()
        self.env.close()
        obs, info = self.env.reset()
        print(f"Episode finished with total reward {total_reward}")
        return log

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
    
    def predict_transformed_action(self, observation: np.ndarray) -> np.ndarray:
        # if last_action is set, this model is actively controlling the environment
        if self.env.unwrapped.last_action is not None:
            action = self.env.unwrapped.last_action.copy()
            self.env.unwrapped.last_action = None
        # if it isn't, predict action again
        else:
            action = self.model.predict(observation[self.observation_indices])[0]
        return self.transform_action(action, observation), []

    def set_callback(self, callback=None) -> None:
        if callback is None:
            self.callback = ModularAgentCallback(self.model_name)
        else:
            self.callback = callback

    def set_model(self):
        model_class, model_args, self.observation_keys = self.get_model_class()
        self.env.unwrapped.create_observation_space(self.observation_keys)
        self.model: Model = model_class(**model_args)
        self.model_name = self.config["model_type"]
        self.observation_indices = self.env.get_wrapper_attr("observation_indices")

    def get_model_class(self):
        if self.config["model_type"] == "PPO":
            model_class = PPO
            model_args = {
                "policy": self.config["policy_type"],
                "env": self.env,
                "learning_rate": self.config["learning_rate"],
                "verbose": 1,
                "seed": self.config["seed"]
            }
            observation_keys = self.config["observation_keys"]
        else:
            model_class = None
            model_args = {"env": self.env}
            for submodule in [policies, contingencies, mixtures, mix2res]:
                for classname in dir(submodule):
                    clss = getattr(submodule, classname)
                    if isinstance(clss, type):
                        if issubclass(clss, Model) and clss is not Model and clss.id == self.config["model_type"]:
                            model_class = clss
                            break
            if model_class is None:
                raise ValueError(f"Model type {self.config['model_type']} not supported for Policy Agent")
            observation_keys = model_class.observation_keys
        return model_class, model_args, observation_keys
    
    @abstractmethod
    def create_action_space(self):
        raise NotImplementedError

    @abstractmethod
    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError
