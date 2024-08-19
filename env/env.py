
import gymnasium as gym
import numpy as np
import pygame


class Environment(gym.Env):
    def __init__(self):
        super().__init__()
        
    def render(self):
        pass
    
    def step(self):
        return self.get_observation(), self.get_reward(), self.get_terminated(), False, self.get_info()
    
    def reset(self):
        return self.get_observation(), self.get_info()
    
    def close(self):
        pygame.quit()
        self.screen = None
        
    def get_observation(self):
        return np.zeros(2)
    
    def get_reward(self):
        return 0.0
    
    def get_terminated(self):
        return False
    
    def get_info():
        return {}