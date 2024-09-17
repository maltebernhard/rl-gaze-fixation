from typing import List
import gymnasium as gym
import numpy as np

from agent.agent import Agent

class MixtureEnv(gym.Env):
    def __init__(self, env: gym.Env, experts, observation_space = None):
        super().__init__()
        self.env = env
        self.experts: List[Agent] = experts
        action_space_dimensionality = 3
        self.action_space = gym.spaces.Box(
            low=np.array([0.0 for _ in range(len(experts)*action_space_dimensionality)]).flatten(),
            high=np.array([1.0 for _ in range(len(experts*action_space_dimensionality))]).flatten(),
            dtype=np.float64
        )
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.observation_space = env.observation_space

    def step(self, action: np.ndarray):
        # n x m array of actions
        actions = self.get_actions(self.observation)
        # save step returns
        self.observation, reward, done, truncated, info = self.env.step(np.sum(action.reshape((len(self.experts),actions[0].shape[0])) * actions, axis = 0))
        return self.observation, reward, done, truncated, info
    
    def reset(self, seed=None, **kwargs):
        self.observation, info = self.env.reset(seed=seed, **kwargs)
        return self.observation, info
    
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)
    
    # =======================================================

    def get_actions(self, observation) -> np.ndarray:
        # n experts, m actions
        # return n x m array
        actions = []
        for expert in self.experts:
            actions.append(expert.predict(observation)[0])
        return np.array(actions)

