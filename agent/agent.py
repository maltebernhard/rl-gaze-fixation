from abc import abstractmethod
from datetime import datetime
from typing import Dict, List
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from agent.model import Model, AvoidNearestObstacleModel, GazeFixationModel, TargetFollowingObstacleEvasionMixtureModel, TowardsTargetModel
from agent.callback import PlottingCallback

# =======================================================    

class StructureAgent:
    def __init__(self, env: gym.Env, agent_config: dict = None) -> None:
        self.env = env
        self.config = agent_config
        self.model: Model = None
        self.model_name = agent_config["model_type"]
        self.set_model()
        self.last_action: np.ndarray = None
        self.set_observation_space(agent_config["observation_keys"])
        self.set_callback()

    def run(self, prints = False, env_seed = None):
        total_reward = 0
        step = 0
        #obs, info = self.env.unwrapped.reset_full_observation(seed=env_seed)
        obs, info = self.env.unwrapped.reset(seed=env_seed)
        if prints:
            print(f'-------------------- Reset ----------------------')
            print(f'Observation: {obs}')
        done = False
        while not done:
            #action, _states = self.predict_full_observation(obs)
            action, _states = self.predict(obs)
            if prints:
                print(f'-------------------- Step {step} ----------------------')
                print(f'Observation: {obs}')
                print(f'Action:      {action}')
            #obs, reward, done, truncated, info = self.env.unwrapped.step_full_observation(action)
            obs, reward, done, truncated, info = self.env.step(action)
            if prints:
                print(f'Reward:      {reward}')
            total_reward += reward
            step += 1
            self.env.render()
            if done:
                #obs, info = self.env.reset_full_observation()
                obs, info = self.env.reset()
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
        print("Observation space: ", [key for key in self.env.get_wrapper_attr("observations").keys()])
        print("Observations: ", self.observation_indices)

    def learn(self, total_timesteps) -> None:
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

    def reset(self):
        set_random_seed(self.config["seed"])

    def save(self, folder = None):
        if folder is None:
            folder = "./training_data/" + datetime.today().strftime('%Y-%m-%d_%H-%M') + "/"
        filename = self.model_name
        self.model.save(folder + filename)

    def load(self, filename):
        if self.model_name == "PPO":
            self.model = PPO.load(filename, env=self.env)

    @abstractmethod
    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, observation: np.ndarray, deterministic = True) -> np.ndarray:
        self.last_action = self.model.predict(observation, deterministic)
        return self.last_action
    
    def predict_full_observation(self, observation: np.ndarray) -> np.ndarray:
        observation = observation[self.observation_indices]
        self.last_action = self.model.predict(observation)
        return self.last_action

    @ abstractmethod
    def set_model(self):
        raise NotImplementedError

    def set_callback(self, callback=None) -> None:
        if callback is None:
            self.callback = PlottingCallback(self.model_name)
        else:
            self.callback = callback
    
# =====================================================================

class Contingency(StructureAgent):
    def __init__(self, base_env, agent_config, contingent_agent) -> None:
        contingency_env = gym.make(
            id='StructureEnv',
            base_env = base_env,
            observation_keys = agent_config["observation_keys"],
            action_space = self.create_action_space()
        )
        self.contingent_agent = contingent_agent
        super().__init__(contingency_env, agent_config)

    def create_action_space(self):
        # TODO: make this more general
        return gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float64,
            shape=(1,)
        )

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        elif self.config["model_type"] == "GFM":
            # TODO: make more general - remove dependency on GazeFixEv
            self.model = GazeFixationModel(self.env, timestep=self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["timestep"], max_vel_rot=self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["robot_max_vel_rot"], max_acc_rot=self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["robot_max_acc_rot"])
        else:
            raise ValueError("Model type not supported for Contingency Agent")

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        previous_action = self.contingent_agent.predict_full_observation(observation)[0]
        self.last_action = np.concatenate([self.contingent_agent.transform_action(previous_action, observation), action])
        return self.last_action

# =======================================================

class Policy(StructureAgent):
    def __init__(self, base_env, agent_config) -> None:
        policy_env = gym.make(
            id='StructureEnv',
            base_env = base_env,
            observation_keys = agent_config["observation_keys"],
            action_space = self.create_action_space()
        )
        super().__init__(policy_env, agent_config)

    def create_action_space(self):
        # TODO: make this more general
        return gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64,
            shape=(2,)
        )

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        # TODO: fix hard-coded action space dimensionality and number of obstacles
        elif self.config["model_type"] == "ANO":
            self.model = AvoidNearestObstacleModel(self.env, action_space_dimensionality=self.config["action_space_dimensionality"], num_obstacles=3)
        elif self.config["model_type"] == "GTT":
            self.model = TowardsTargetModel(self.env, action_space_dimensionality=self.config["action_space_dimensionality"])
        else:
            raise ValueError("Model type not supported for Policy Agent")

    def transform_action(self, action, observation) -> np.ndarray:
        self.last_action = action
        return self.last_action

# =======================================================

class MixtureOfExperts(StructureAgent):
    def __init__(self, base_env, agent_config, experts):
        mixture_env = gym.make(
            id='StructureEnv',
            base_env = base_env,
            observation_keys = agent_config["observation_keys"],
            action_space = self.create_action_space(agent_config["mixture_mode"], len(experts))
        )
        super().__init__(mixture_env, agent_config)
        self.experts: List[StructureAgent] = experts
        self.mixture_mode = agent_config["mixture_mode"]

    def create_action_space(self, mixture_mode, num_experts):
        if mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif mixture_mode == 2:
            self.action_space_dimensionality = self.base_env.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        return gym.spaces.Box(
            low=np.array([0.0 for _ in range(num_experts*self.action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(num_experts*self.action_space_dimensionality)]).flatten(),
            dtype=np.float64
        )

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        elif self.config["model_type"] == "TOM":
            self.model = TargetFollowingObstacleEvasionMixtureModel(self.env, mixture_mode=self.config["mixture_mode"], action_space_dimensionality=3)
        else:
            raise ValueError("Model type not supported for Mixture-of-Experts Agent")

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        weights = self.normalize_weights(action)
        # get mixture experts' actions
        actions = []
        # n x m array of actions
        for expert in self.experts:
            a = expert.predict_full_observation(observation)[0]
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
        self.last_agent: StructureAgent = last_agent

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.last_agent.transform_action(self.last_agent.predict_full_observation(observation)[0], observation)