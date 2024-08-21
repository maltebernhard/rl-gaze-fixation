import numpy as np
from typing import List
import gymnasium as gym
from contingency.contingency import GazeFixation
from env.env import Environment

class Agent(gym.Env):
    def __init__(self, timestep: float, action_mode: int, use_contingencies: bool, distance: float):
        super().__init__()
        self.env : Environment = gym.make(id='GazeFixEnv',
                                          timestep = timestep,
                                          action_mode = action_mode,
                                          distance = distance)
        self.action_mode = action_mode
        self.timestep = timestep

        self.metadata = self.env.metadata

        self.use_contingencies = use_contingencies
        self.contingencies = [GazeFixation(self.env.robot.max_acc_phi, self.action_mode)]
        
        self.observation_space = self.observation_space = gym.spaces.Box(low=np.array([-self.env.robot.sensor_angle/2, -self.env.robot.max_vel_phi]), high=np.array([self.env.robot.sensor_angle/2, self.env.robot.max_vel_phi]), shape=(2,))

        if self.action_mode == 1:
            if use_contingencies:
                self.action_space = gym.spaces.Box(
                    low=np.array([-self.env.robot.max_acc]*2),
                    high=np.array([self.env.robot.max_acc]*2),
                    shape=(2,),
                    dtype=np.float64
                )
            else:
                self.action_space = gym.spaces.Box(
                    low=np.array(([-self.env.robot.max_acc, -self.env.robot.max_acc, -self.env.robot.max_acc_phi])),
                    high=np.array(([self.env.robot.max_acc, self.env.robot.max_acc, self.env.robot.max_acc_phi])),
                    shape=(3,),
                    dtype=np.float64
                )
        elif self.action_mode == 2:
            if use_contingencies:
                self.action_space = gym.spaces.MultiDiscrete(
                    np.array([[3, 3]])
                )
            else:
                self.action_space = gym.spaces.MultiDiscrete(
                    np.array([[3, 3, 3]])
                )
        else:
            raise NotImplementedError
        

        self.history: List[List[float]] = []
        self.history_len = 2
        self.total_reward = 0.0

    def step(self, action):
        if self.use_contingencies:
            for c in self.contingencies:
                action = c.contingent_action(self.state, action)
        obs, reward, done, truncated, info = self.env.step(action)
        self.total_reward += reward
        self._get_state(obs)
        return self.state, reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        self.total_reward = 0.0
        obs, info = self.env.reset(seed=seed, **kwargs)
        return self._get_state(obs), info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()

    def _get_state(self, observation):
        # add observation to history
        if len(self.history) == self.history_len:
            self.history.pop()
        self.history.insert(0,observation)
        # TODO: better implementation - currently: add first element twice
        if len(self.history) == 1:
            self.history.insert(0,observation)
        vel = (self.history[1][1][0]-self.history[0][1][0])/self.timestep
        if observation[0] == 1:
            self.state = np.concatenate([observation[1], np.array([vel])])
        else:
            self.state = np.array([np.pi, 0.0])
        return self.state
            
    def env_attr(self, attr):
        return self.env.get_wrapper_attr(attr)