import numpy as np
from environment.env import Environment
from training_logging.plotting import PlottingCallback

class AvoidNearestObstacleModel:
    def __init__(self, env: Environment, action_space_dimensionality: int = 3):
        self.env = env
        self.num_obstacles = self.env.unwrapped.config["num_obstacles"]
        #self.observation_keys = [item for o in range(self.num_obstacles) for item in [f"obstacle{o+1}_offset_angle", f"obstacle{o+1}_coverage", f"obstacle{o+1}_distance"]]
        self.observation_keys = [item for o in range(self.num_obstacles) for item in [f"obstacle{o+1}_offset_angle", f"obstacle{o+1}_coverage"]]
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