import math

import numpy as np
from environment.agent import Agent

from plotting.plotting import PlottingCallback

class BaselineModel:
    def __init__(self, env: Agent):
        self.env = env
        self.action_mode = self.env.action_mode
        self.use_contingencies = self.env.use_contingencies
        self.state = None
        self.action = None

        self.max_acc = self.env.unwrapped.env.unwrapped.robot.max_acc
        
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
        self.state = state

        acc_lateral = 1
        if self.state[-2] > eps: acc_frontal = 2
        elif self.state[-2] < 0:
            acc_frontal = 1
            if self.state[-4] > 0:
                acc_lateral = 0
        else:
            acc_frontal = 1
            if self.state[-4] > 0:
                acc_lateral = 0
            elif self.state[-4] < 0:
                acc_frontal = 2

        if self.use_contingencies: self.action = np.array([acc_frontal, acc_lateral])
        else: self.action = np.array([acc_frontal, acc_lateral, 1])
        if self.action_mode == 1:
            self.action = (self.action - np.ones(self.action.shape)) * self.max_acc
        return self.action, []
        
    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass