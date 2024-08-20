import math
from env.agent import Agent

from plotting.plotting import PlottingCallback

class BaselineModel:
    def __init__(self, env: Agent):
        self.env = env
        self.state = None
        self.action = None
        
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
        
    def predict(self, state, deterministic: bool = True):
        if self.state is None:
            self.state = state
        if self.action is None:
            self.action = [2.0,0.0]
        
        return self.action, []
        
    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass