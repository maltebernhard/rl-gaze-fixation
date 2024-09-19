from abc import abstractmethod
from typing import Dict, List
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from model.model import Model
from training_logging.plotting import PlottingCallback, plot_training_progress

# =======================================================

class Agent:
    def __init__(self, env: gym.Env, agent_config: dict = None) -> None:
        self.env = env
        # TODO: make config dict
        self.config = agent_config
        self.model: Model = None

    def run(self, prints = False, env_seed = None):
        total_reward = 0
        step = 0
        self.env.reset(seed=env_seed)
        obs, info = self.env.unwrapped.reset_full_observation(seed=env_seed)
        if prints:
            print(f'-------------------- Reset ----------------------')
            print(f'Observation: {obs}')
        done = False
        while not done:
            action, _states = self.predict(obs)
            if prints:
                print(f'-------------------- Step {step} ----------------------')
                print(f'Observation: {obs}')
                print(f'Action:      {action}')
            obs, reward, done, truncated, info = self.env.unwrapped.step_full_observation(action)
            if prints:
                print(f'Reward:      {reward}')
            total_reward += reward
            step += 1
            self.env.render()
            if done:
                obs, info = self.env.reset_full_observation()
        print(f"Episode finished with total reward {total_reward}")

    def set_observation_space(self, observation_keys: List[str] = None) -> None:
        if observation_keys is None:
            self.observation_indices = self.env.get_wrapper_attr("observation_indices")
            return
        index = 0
        observation_indices = []
        for key in self.env.get_wrapper_attr("observations").keys():
            if key in observation_keys:
                observation_indices.append(index)
            index += 1
        self.observation_indices = np.array(observation_indices)

    def learn(self, total_timesteps) -> None:
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

    def save(self, name) -> None:
        self.model.save(name)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        observation = observation[self.observation_indices]
        return self.model.predict(observation)
    
# =======================================================

class Contingency(Agent):
    def __init__(self, base_env, contingent_agent, observation_keys, agent_config = None) -> None:
        contingency_env = gym.make(
            id='ContingencyEnv',
            base_env = base_env,
            agent = contingent_agent,
            observation_keys = observation_keys,
        )
        super().__init__(contingency_env, agent_config)
        self.model = None
    
    def set_model(self, model) -> None:
        self.model = model

# =======================================================

class Policy(Agent):
    def __init__(self, base_env, observation_keys, agent_config = None) -> None:
        policy_env = gym.make(
            id='PolicyEnv',
            base_env = base_env,
            observation_keys = observation_keys,
        )
        super().__init__(policy_env, agent_config)
        self.model = None

    def set_model(self, model = None, seed=1) -> None:
        if model is None:
            self.model = PPO("MlpPolicy", self.env, learning_rate=0.0003, verbose=1, seed=seed)
        else:
            self.model = model

# =======================================================

class MixtureOfExperts(Agent):
    def __init__(self, base_env, observation_keys, experts, mixture_mode = 1, agent_config = None):
        mixture_env = gym.make(
            id='MixtureEnv',
            base_env = base_env,
            observation_keys = observation_keys,
            experts = experts,
            mixture_mode = mixture_mode,
        )
        super().__init__(mixture_env, agent_config)
        self.experts: List[Agent] = experts
        self.mixture_mode = mixture_mode
        self.callback = None

    def set_callback(self, callback=None) -> None:
        if callback is None:
            self.callback = PlottingCallback("MixtureOfExperts")
        else:
            self.callback = callback

    def set_model(self, model = None, seed=1) -> None:
        if model is None:
            self.model = PPO("MlpPolicy", self.env, learning_rate=0.0003, verbose=1, seed=seed)
        else:
            self.model = model