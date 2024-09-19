from datetime import datetime
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.utils import set_random_seed
from environment.structure_env import StructureEnv
from training_logging.plotting import PlottingCallback

# ==========================================================================

class ModelWrapper:
    def __init__(self, env, config={}) -> None:
        self.env = env
        self.set_model(config)
        # Create the callback
        self.callback = PlottingCallback(self.model_name)

    def set_model(self, config):
        if config["policy_type"] == "PPO":
            self.model = PPO(self.config["policy_type"], self.env, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        elif config["policy_type"] == "ANO":
            self.model = AvoidNearestObstacleModel(self.env, action_space_dimensionality=3)
        elif config["policy_type"] == "GTT":
            self.model = TowardsTargetModel(self.env, action_space_dimensionality=2)
        elif config["policy_type"] == "TOM":
            self.model = TargetFollowingObstacleEvasionMixtureModel(self.env, mixture_mode=self.config["mixture_mode"], action_space_dimensionality=3)
            
        self.config = config
        self.set_model_name()

    def set_model_name(self):
        self.model_name = "PPO"

    def learn(self, total_timesteps, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, obs, deterministic = True):
        return self.model.predict(obs, deterministic)
    
    def reset(self):
        set_random_seed(self.config["seed"])

    def save(self, folder = None):
        if folder is None:
            folder = "./training_data/" + datetime.today().strftime('%Y-%m-%d_%H-%M') + "/"
        filename = self.model_name
        self.model.save(folder + filename)

    def load(self, filename, env):
        self.model = PPO.load(filename, env=env)

# =============================================================================

class Model:
    def __init__(self, env: StructureEnv, config={}) -> None:
        self.env = env

# =============================================================================

class GazeFixationModel(Model):
    def __init__(self, env: StructureEnv, timestep, max_vel_rot, max_acc_rot):
        super().__init__(env)
        self.max_vel = max_vel_rot
        self.max_acc = max_acc_rot
        self.angle = (self.max_vel ** 2) / (2 * self.max_acc)
        self.action_mode = 1
        self.epsilon = timestep * 0.5

    def predict(self, obs, deterministic = True):
        vel_rot_desired = self.compute_target_vel(obs[0])
        if self.action_mode == 1:
            return np.array([self.pd_control(vel_rot_desired-obs[1], obs[1])]), []
        elif self.action_mode == 2:
            return np.array([self.flip_control(vel_rot_desired, obs[1], self.epsilon)]), []
        
    def flip_control(self, target, current, eps):
        action = 1
        if target-current > eps:
            action = 2
        elif target-current < -eps:
            action = 0
        if self.action_mode == 1:
            return (action-1)
        return action

    # TODO: only works for continuous action
    def pd_control(self, x, del_x):
        K_p = 1.0
        K_d = 0.1
        return K_p * x - K_d * del_x
    
    def p_control(self, x):
        K_p = 10.0
        return K_p * x
    
    def compute_target_vel(self, angle):
        if abs(angle) > self.angle:
            vel_rot_desired = angle/abs(angle)
        else:
            vel_rot_desired = angle/self.angle
        return vel_rot_desired

# =============================================================================

class AvoidNearestObstacleModel(Model):
    def __init__(self, env: StructureEnv, action_space_dimensionality: int = 3):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.config["num_obstacles"]
        self.action_space_dimensionality = action_space_dimensionality
        
    def learn(self, total_timesteps: int, callback: PlottingCallback):
        obs, info = self.env.reset()
        timestep = 0
        while timestep < total_timesteps:
            total_reward = 0
            done = False
            while not done:
                action, _states = self.predict(obs, deterministic=True)
                # print(f'Observation: {obs} | Action: {action}')
                obs, reward, done, truncated, info = self.env.step(action)
                callback.locals['rewards'] = [reward]
                callback.locals['dones'] = [done]
                callback._on_step()
                timestep += 1
                total_reward += reward
                if done:
                    obs, info = self.env.reset()
            print("Episode reward: {}".format(total_reward))

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        nearest_obstacle = 0
        for i in range(1,self.num_obstacles):
            if state[2*i+1] > state[2*nearest_obstacle+1]:
                nearest_obstacle = i
        offset_angle = state[2*nearest_obstacle]
        # create a vector of length one in opposite direction of the obstacle
        if self.action_space_dimensionality == 3:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        elif self.action_space_dimensionality == 2:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, []
        
    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass

# =============================================================================

class TowardsTargetModel(Model):
    def __init__(self, env: StructureEnv, action_space_dimensionality: int = 3):
        super().__init__(env)
        self.action_space_dimensionality = action_space_dimensionality
        
    def learn(self, total_timesteps: int, callback: PlottingCallback):
        obs, info = self.env.reset()
        timestep = 0
        while timestep < total_timesteps:
            total_reward = 0
            done = False
            while not done:
                action, _states = self.predict(obs, deterministic=True)
                # print(f'Observation: {obs} | Action: {action}')
                obs, reward, done, truncated, info = self.env.step(action)
                callback.locals['rewards'] = [reward]
                callback.locals['dones'] = [done]
                callback._on_step()
                timestep += 1
                total_reward += reward
                if done:
                    obs, info = self.env.reset()
            print("Episode reward: {}".format(total_reward))

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        offset_angle = state[0]
        # generate unit length vector in the direction of the target
        if self.action_space_dimensionality == 2:
            action = np.array([np.cos(offset_angle), np.sin(offset_angle)])
        elif self.action_space_dimensionality == 3:
            action = np.array([np.cos(offset_angle), np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, []
        
    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass

# =============================================================================

class TargetFollowingObstacleEvasionMixtureModel(Model):
    def __init__(self, env: StructureEnv, mixture_mode: int = 1, action_space_dimensionality: int = 3):
        super().__init__(env)
        self.mixture_mode = mixture_mode
        self.action_space_dimensionality = action_space_dimensionality
        
    def learn(self, total_timesteps: int, callback: PlottingCallback):
        obs, info = self.env.reset()
        timestep = 0
        while timestep < total_timesteps:
            total_reward = 0
            done = False
            while not done:
                action, _states = self.predict(obs, deterministic=True)
                # print(f'Observation: {obs} | Action: {action}')
                obs, reward, done, truncated, info = self.env.step(action)
                callback.locals['rewards'] = [reward]
                callback.locals['dones'] = [done]
                callback._on_step()
                timestep += 1
                total_reward += reward
                if done:
                    obs, info = self.env.reset()
            print("Episode reward: {}".format(total_reward))

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        # for states with 3 obstacles, represented by angular offset and fov coverage each
        obstacle = max(max(state[-1], state[-3]), state[-5])
        relevance_obstacle_evasion = min(obstacle**2 * 15, 1.0)
        if self.mixture_mode == 1:
            weights = np.array([[relevance_obstacle_evasion], [1.0 - relevance_obstacle_evasion]])
        elif self.mixture_mode == 2:
            if self.action_space_dimensionality == 3:
                weights = np.array(
                    [
                        [relevance_obstacle_evasion,        relevance_obstacle_evasion,       1.0],
                        [1.0 - relevance_obstacle_evasion,  1.0 - relevance_obstacle_evasion, 1.0]
                    ]
                )
            elif self.action_space_dimensionality == 2:
                weights = np.array(
                    [
                        [relevance_obstacle_evasion,        relevance_obstacle_evasion],
                        [1.0 - relevance_obstacle_evasion,  1.0 - relevance_obstacle_evasion]
                    ]
                )
            else:
                raise Exception("Action space dimensionality not supported.")
        else:
            raise Exception("Mixture mode not supported.")
        return weights.flatten(), []
        
    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass

# =============================================================================

