import numpy as np
from agent.base_model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class GazeFixationModel(Model):
    id = "GFM"
    observation_keys = ["target_offset_angle", "del_target_offset_angle"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.max_vel = env.unwrapped.base_env_config["robot_max_vel_rot"]
        self.max_acc = env.unwrapped.base_env_config["robot_max_acc_rot"]
        self.breaking_angle = (self.max_vel ** 2) / (2 * self.max_acc)
        self.action_mode = env.unwrapped.base_env_config["action_mode"]
        self.timestep = env.unwrapped.base_env_config["timestep"]
        self.epsilon = self.timestep

    def predict(self, obs, deterministic = True):
        vel_rot_desired = self.compute_target_vel(obs[0])
        if self.action_mode == 1:
            return np.array([self.pd_control(vel_rot_desired-obs[1], obs[1])]), None
        elif self.action_mode == 2:
            return np.array([self.flip_control(vel_rot_desired, obs[1])]), None
        elif self.action_mode == 3:
            return np.array([self.vel_control(obs[0])]), None
        
    def vel_control(self, target_offset_angle):
        return min(target_offset_angle/(self.max_vel*self.timestep), 1.0)

    def flip_control(self, target, current):
        action = 1
        if target-current > self.epsilon:
            action = 2
        elif target-current < -self.epsilon:
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
    
    def compute_target_vel(self, target_angle_offset):
        if abs(target_angle_offset) > self.breaking_angle:
            vel_rot_desired = target_angle_offset/abs(target_angle_offset)
        else:
            vel_rot_desired = target_angle_offset/self.breaking_angle
        return vel_rot_desired