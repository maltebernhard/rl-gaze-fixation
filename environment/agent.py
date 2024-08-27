import numpy as np
from typing import List
import gymnasium as gym
from contingency.contingency import GazeFixation
from environment.env import Environment

class Agent(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.env : Environment = gym.make(id='GazeFixEnv',
                                          config = config)
        
        self.config = config

        self.timestep: float = config["timestep"]
        self.action_mode: int = config["action_mode"]
        self.target_distance: float = config["target_distance"]
        self.wall_collision: bool = config["wall_collision"]
        self.obstacles: bool = config["num_obstacles"]

        self.metadata = self.env.metadata

        self.use_contingencies = config["use_contingencies"]
        self.contingencies = [GazeFixation(self.timestep, self.config["robot_max_vel_rot"], self.config["robot_max_acc_rot"], self.action_mode)]
        
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.config["robot_sensor_angle"]/2, -self.config["robot_max_vel_rot"], -self.config["robot_max_vel"], -self.config["robot_max_vel"], -self.config["target_distance"], -np.inf]),
            high=np.array([self.config["robot_sensor_angle"]/2, self.config["robot_max_vel_rot"], self.config["robot_max_vel"], self.config["robot_max_vel"], np.inf, np.inf]),
            shape=(6,),
            dtype=np.float64
        )

        if self.action_mode == 1:
            if self.use_contingencies:
                self.action_space = gym.spaces.Box(
                    low=np.array([-self.config["robot_max_acc"]]*2),
                    high=np.array([self.config["robot_max_acc"]]*2),
                    shape=(2,),
                    dtype=np.float32
                )
            else:
                self.action_space = gym.spaces.Box(
                    low=np.array(([-self.config["robot_max_acc"], -self.config["robot_max_acc"], -self.config["robot_max_acc_rot"]])),
                    high=np.array(([self.config["robot_max_acc"], self.config["robot_max_acc"], self.config["robot_max_acc_rot"]])),
                    shape=(3,),
                    dtype=np.float32
                )
        elif self.action_mode == 2:
            if self.use_contingencies:
                self.action_space = gym.spaces.MultiDiscrete(
                    np.array([3, 3])
                )
            else:
                self.action_space = gym.spaces.MultiDiscrete(
                    np.array([3, 3, 3])
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
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
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
        if len(self.history) == 1:
            self.history.insert(0,observation)

        # relative rotational velocity of target
        vel_rot = (self.history[1][0]-self.history[0][0])/self.timestep
        self.state = np.concatenate([observation, np.array([vel_rot])])

        return self.state
            
    def env_attr(self, attr):
        return self.env.get_wrapper_attr(attr)