from abc import abstractmethod
from datetime import datetime
from typing import List
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from agent.models.model import Model
from agent.models.avoid_nearest_obstacle import AvoidNearestObstacleModel
from agent.models.towards_target import TowardsTargetModel
from agent.models.gaze_fixation import GazeFixationModel
from agent.models.target_following_obstacle_evasion_mixture import TargetFollowingObstacleEvasionMixtureModel
from utils.callback import PlottingCallback, ModularAgentCallback
from environment.structure_env import StructureEnv

# =======================================================    

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
    
# ===================================================================================

class Policy(StructureAgent):
    def __init__(self, base_env, agent_config, callback) -> None:
        super().__init__(base_env, agent_config, callback)

    def create_action_space(self):
        # TODO: make this more general
        return gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64,
            shape=(2,)
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

# =========================================================================

class Contingency(StructureAgent):
    def __init__(self, base_env, agent_config, callback, contingent_agent) -> None:
        self.contingent_agent: StructureAgent = contingent_agent
        super().__init__(base_env, agent_config, callback)

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
            self.model = GazeFixationModel(self.env, timestep=self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["timestep"], max_vel_rot=self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["robot_max_vel_rot"], max_acc_rot=self.env.unwrapped.base_env.unwrapped.env.unwrapped.config["robot_max_acc_rot"])
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

# =========================================================================================================

class MixtureOfExperts(StructureAgent):
    def __init__(self, base_env, agent_config, callback, experts):
        self.experts: List[StructureAgent] = experts
        self.mixture_mode = agent_config["mixture_mode"]
        super().__init__(base_env, agent_config, callback)

    def create_action_space(self):
        if self.mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif self.mixture_mode == 2:
            self.action_space_dimensionality = self.env.unwrapped.base_env.unwrapped.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        return gym.spaces.Box(
            low=np.array([0.0 for _ in range(len(self.experts)*self.action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(len(self.experts)*self.action_space_dimensionality)]).flatten(),
            dtype=np.float64
        )

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        else:
            raise ValueError("Model type not supported for Mixture-of-Experts Agent")
        
    def get_observation_keys(self):
        if self.config["model_type"] == "PPO":
            return self.config["observation_keys"]
        else:
            return None

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        weights = self.normalize_weights(action)
        # get mixture experts' actions
        actions = []
        # n x m array of actions
        for expert in self.experts:
            a = expert.predict_full_observation(observation)[0]
            actions.append(expert.transform_action(a, observation))
        # apply mixture
        action = np.sum(weights * actions, axis = 0)
        return action
    
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

# =========================================================================================================

class MixtureOfTwoExperts(StructureAgent):
    def __init__(self, base_env, agent_config, callback, experts):
        self.experts: List[StructureAgent] = experts
        self.mixture_mode = agent_config["mixture_mode"]
        super().__init__(base_env, agent_config, callback)

    def create_action_space(self):
        if self.mixture_mode == 1:
            self.action_space_dimensionality = 1
        elif self.mixture_mode == 2:
            self.action_space_dimensionality = self.experts[0].env.action_space.shape[0]
        else:
            raise ValueError("Mixture mode not supported")
        return gym.spaces.Box(
            low=np.array([0.0 for _ in range(self.action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(self.action_space_dimensionality)]).flatten(),
            dtype=np.float64
        )

    def set_model(self):
        if self.config["model_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        elif self.config["model_type"] == "TOM":
            self.model = TargetFollowingObstacleEvasionMixtureModel(self.env, mixture_mode=self.config["mixture_mode"], action_space_dimensionality=self.action_space_dimensionality)
        else:
            raise ValueError("Model type not supported for Mixture-of-Experts Agent")
        
    def get_observation_keys(self):
        if self.config["model_type"] == "PPO":
            return self.config["observation_keys"]
        elif self.config["model_type"] == "TOM":
            return TargetFollowingObstacleEvasionMixtureModel.observation_keys
        else:
            return None

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = np.array([[action[0]], [1.0 - action[0]]])
            weights = weights.reshape((len(self.experts),1)).repeat(self.action_space_dimensionality, axis=1)
        elif self.mixture_mode == 2:
            weights = np.array([action, np.ones(action.shape) - action])
        # get mixture experts' actions
        actions = []
        # n x m array of actions
        for expert in self.experts:
            a = expert.predict_full_observation(observation)[0]
            actions.append(expert.transform_action(a, observation))
        # apply mixture
        action = np.sum(weights * actions, axis = 0)
        return action