from abc import abstractmethod
from typing import Dict, List
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from contingency.contingency import SMC
from environment.env import Observation
from training_logging.plotting import PlottingCallback, plot_training_progress

# =======================================================

class Agent:
    def __init__(self, env: gym.Env, agent_config: dict = None) -> None:
        self.env: gym.Env = env
        # TODO: make config dict
        self.config = agent_config
        self.observation_keys = None
        self.observation_indices = None
        self.observation_space = None
        self.action_space = None

    def run(self, prints = False, env_seed = None):
        total_reward = 0
        step = 0
        obs, info = self.env.reset(seed=env_seed)
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
            obs, reward, done, truncated, info = self.env.step(action)
            if prints:
                print(f'Reward:      {reward}')
            total_reward += reward
            step += 1
            self.env.render()
            if done:
                obs, info = self.env.reset()
        print(f"Episode finished with total reward {total_reward}")

    def set_observation_space(self, observation_keys: List[str] = None) -> None:
        if (self.observation_keys is None) and (observation_keys is not None):
            self.observation_keys = observation_keys
        elif (self.observation_keys is None) and (observation_keys is None):
            self.observation_indices = np.array([i for i in range(self.env.observation_space.shape[0])])
            return
        observation_dict: Dict[str, Observation] = {}
        index = 0
        for key, obs in self.env.unwrapped.observations.items():
            if key in self.observation_keys:
                observation_dict[key] = (obs, index)
            index += 1
        self.observation_indices = np.array([obs[1] for obs in observation_dict.values()])
        self.observation_space = gym.spaces.Box(
            low=np.array([obs[0].low for obs in observation_dict.values()]).flatten(),
            high=np.array([obs[0].high for obs in observation_dict.values()]).flatten(),
            dtype=np.float64
        )

    def set_action_space(self, action_space = None):
        if self.action_space is None:
            self.action_space = self.env.action_space
        else:
            self.action_space = action_space

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
# =======================================================

class Contingency(Agent):
    def __init__(self, env, contingent_agent, smc = None, agent_config = None) -> None:
        super().__init__(env, agent_config)
        self.c_agent: Agent = contingent_agent
        self.smc: SMC = smc

    def predict(self, observation) -> np.ndarray:
        action = self.c_agent.predict(observation)[0]
        observation = observation[self.observation_indices]
        action = self.smc.contingent_action(observation, action)
        return action, []
    
    def set_smc(self, smc) -> None:
        self.smc = smc

# =======================================================

class Policy(Agent):
    def __init__(self, env, agent_config = None) -> None:
        super().__init__(env, agent_config)
        self.model = None

    def predict(self, observation) -> np.ndarray:
        observation = observation[self.observation_indices]
        action = np.ones(self.action_space.shape)
        if self.model is not None:
            action = self.model.predict(observation)[0]
        return action, []

    # def set_model(self, model_config) -> None:
    #     self.model = PPO(model_config["policy_type"], self.env, learning_rate=model_config["learning_rate"], verbose=1, seed=model_config["seed"])
    
    def set_model(self, model) -> None:
        self.model = model
        self.observation_keys = self.model.observation_keys

# =======================================================

class MixtureOfExperts(Agent):
    def __init__(self, env, experts, agent_config = None):
        super().__init__(env, agent_config)
        self.mixture_env = gym.make(id='MixtureEnv',
                                    env = env,
                                    experts = experts,
                                   )
        self.experts: List[Agent] = experts
        self.model = None
        self.callback = None

    def predict(self, observation) -> np.ndarray:
        # n x m array of actions
        actions = self.get_actions(observation)
        # reduce observation to relevant features
        observation = observation[self.observation_indices]
        # one-dimensional n*m array of weights
        weights = self.get_weights(observation)
        # return weighted sum of actions with reshaped weights array
        return np.sum(weights.reshape((len(self.experts),self.action_space.shape[0])) * actions, axis = 0), []
    
    def get_actions(self, observation) -> np.ndarray:
        return self.mixture_env.unwrapped.get_actions(observation)

    def get_weights(self, observation) -> np.ndarray:
        if self.model is None:
            obstacle = max(max(observation[-1], observation[-3]),observation[-5])
            relevance_obstacle_evasion = min(obstacle**2 * 10, 1.0)
            weights = np.array(
                [
                    [relevance_obstacle_evasion,        relevance_obstacle_evasion,       1.0],
                    [1.0 - relevance_obstacle_evasion,  1.0 - relevance_obstacle_evasion, 1.0]
                ]
            )
        else:
            weights = self.model.predict(observation)[0].reshape((len(self.experts),self.action_space.shape[0]))
        # normalize weights
        sum = np.sum(weights, axis=0)
        weights[:, sum == 0] = 1 / weights.shape[0]
        sum = np.sum(weights, axis=0)
        weights = weights / sum

        return weights.flatten()
    
    def set_model(self) -> None:
        self.model = PPO("MlpPolicy", self.mixture_env, learning_rate=0.0003, verbose=1, seed=140)
        self.callback = PlottingCallback("MixtureOfExperts")

    def learn(self, total_timesteps) -> None:
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        plot_training_progress(self.callback)
        self.model.save("mixture_model")