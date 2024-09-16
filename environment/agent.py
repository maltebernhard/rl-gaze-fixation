import numpy as np
from typing import Dict, List
import gymnasium as gym
from contingency.contingency import GazeFixation
from environment.env import Environment, Observation

class Agent(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.env : Environment = gym.make(id='GazeFixEnv',
                                          config = config)
        self.config = config

        self.timestep: float = config["timestep"]
        self.observe_distance: float = config["observe_distance"]
        self.action_mode: int = config["action_mode"]
        self.max_target_distance: float = self.env.unwrapped.max_target_distance
        self.wall_collision: bool = config["wall_collision"]
        self.num_obstacles: bool = config["num_obstacles"]

        self.metadata = self.env.metadata

        self.use_contingencies = config["use_contingencies"]
        self.contingencies = [GazeFixation(self.timestep, self.config["robot_max_vel_rot"], self.config["robot_max_acc_rot"], self.action_mode)]
        
        self.observations: Dict[str, Observation] = self.env.unwrapped.observations.copy()

        if self.use_contingencies:
            self.observations.pop("target_offset_angle")
            self.observations.pop("del_target_offset_angle")

        self.observation_space = gym.spaces.Box(
            low=np.array([obs.low for obs in self.observations.values()]),
            high=np.array([obs.high for obs in self.observations.values()]),
            shape=(len(self.observations),),
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
        state = self._get_state(obs)
        return state, reward, done, truncated, info

    def reset(self, seed=None, record_video=False, video_path = "", **kwargs):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        self.total_reward = 0.0
        obs, info = self.env.reset(seed=seed, record_video=record_video, video_path=video_path, **kwargs)
        return self._get_state(obs), info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()

    def _get_state(self, observation):
        # add observation to history
        while len(self.history) < self.history_len + 1:
            self.history.insert(0,observation)
        self.history.pop()

        self.state = observation

        if self.use_contingencies:
            return np.concatenate([self.state[:1], self.state[3:]])
        else:
            return self.state
            
    def env_attr(self, attr):
        return self.env.get_wrapper_attr(attr)