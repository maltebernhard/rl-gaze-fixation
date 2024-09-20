from abc import abstractmethod
from typing import Dict, List
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from model.model import Model
from training_logging.plotting import PlottingCallback

# =======================================================    

class StructureAgent:
    def __init__(self, env: gym.Env, observation_keys: List[str], agent_config: dict = None) -> None:
        self.env = env
        self.config = agent_config
        self.model: Model = None
        self.last_action: np.ndarray = None
        self.set_observation_space(observation_keys)
        self.set_callback()

    def run(self, prints = False, env_seed = None):
        total_reward = 0
        step = 0
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
            #obs, reward, done, truncated, info = self.env.step(action)
            if prints:
                print(f'Reward:      {reward}')
            total_reward += reward
            step += 1
            self.env.render()
            if done:
                obs, info = self.env.reset_full_observation()
                #obs, info = self.env.reset()
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

    @abstractmethod
    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, observation: np.ndarray) -> np.ndarray:
        observation = observation[self.observation_indices]
        self.last_action = self.model.predict(observation)
        return self.last_action
    
    def set_model(self, model) -> None:
        self.model = model

    def set_callback(self, callback=None) -> None:
        if callback is None:
            self.callback = PlottingCallback("MixtureOfExperts")
        else:
            self.callback = callback
    
# =======================================================

class Contingency(StructureAgent):
    def __init__(self, base_env, contingent_agent, observation_keys, agent_config = None) -> None:
        contingency_env = gym.make(
            id='ContingencyEnv',
            base_env = base_env,
            contingent_agent = contingent_agent,
            observation_keys = observation_keys,
        )
        self.contingent_agent = contingent_agent
        super().__init__(contingency_env, observation_keys, agent_config)
        self.model = None

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        previous_action = self.contingent_agent.predict(observation)[0]
        self.last_action = np.concatenate([self.contingent_agent.transform_action(previous_action, observation), action])
        return self.last_action

# =======================================================

class Policy(StructureAgent):
    def __init__(self, base_env, observation_keys, agent_config = None) -> None:
        policy_env = gym.make(
            id='PolicyEnv',
            base_env = base_env,
            observation_keys = observation_keys,
        )
        super().__init__(policy_env, observation_keys, agent_config)
        self.model = None

    def transform_action(self, action, observation) -> np.ndarray:
        self.last_action = action
        return self.last_action

# =======================================================

class MixtureOfExperts(StructureAgent):
    def __init__(self, base_env, observation_keys, experts, mixture_mode = 1, agent_config = None):
        mixture_env = gym.make(
            id='MixtureEnv',
            base_env = base_env,
            observation_keys = observation_keys,
            experts = experts,
            mixture_mode = mixture_mode,
        )
        super().__init__(mixture_env, observation_keys, agent_config)
        self.experts: List[StructureAgent] = experts
        self.mixture_mode = mixture_mode
        self.callback = None

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        weights = self.normalize_weights(action)
        # get mixture experts' actions
        actions = []
        # n x m array of actions
        for expert in self.experts:
            a = expert.predict(observation)[0]
            actions.append(expert.transform_action(a, observation))
        # apply mixture
        self.last_action = np.sum(weights * actions, axis = 0)
        return self.last_action
    
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = weights.reshape((len(self.experts),1)).repeat(self.experts[0].env.action_space.shape[0], axis=1)
        else:
            weights = weights.reshape((len(self.experts),self.experts[0].env.action_space.shape[0]))
        # prevent 0-weights
        sum = np.sum(weights, axis=0)
        weights[:, sum == 0] = 1 / weights.shape[0]
        sum = np.sum(weights, axis=0)
        # normalize weights
        weights = weights / sum
        return weights

# =======================================================================================

class BaseAgent:
    def __init__(self, agents: List[StructureAgent]) -> None:
        self.agents = agents
        self.last_agent = None

    def set_last_agent(self, last_agent) -> None:
        self.last_agent = last_agent

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.last_agent.transform_action(self.last_agent.predict(observation)[0], observation)