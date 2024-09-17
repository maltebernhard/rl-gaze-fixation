import numpy as np
from environment.env import Environment
from training_logging.plotting import PlottingCallback

class TowardsTargetModel:
    def __init__(self, env: Environment, action_space_dimensionality: int = 3):
        self.env = env
        self.observation_keys = ["target_offset_angle", "del_target_offset_angle", "vel_rot", "vel_frontal", "vel_lateral"]
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