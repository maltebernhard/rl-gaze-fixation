import numpy as np
from agent.models.model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class GazeFixationModel(Model):
    observation_keys = ["target_offset_angle", "del_target_offset_angle"]

    def __init__(self, env: StructureEnv, timestep, max_vel_rot, max_acc_rot):
        super().__init__(env)
        self.max_vel = max_vel_rot
        self.max_acc = max_acc_rot
        self.angle = (self.max_vel ** 2) / (2 * self.max_acc)
        self.action_mode = 1
        self.epsilon = timestep * 0.5

    def predict(self, obs, deterministic = True):
        vel_rot_desired = self.compute_target_vel(obs[0])
        if self.action_mode == 1:
            return np.array([self.pd_control(vel_rot_desired-obs[1], obs[1])]), None
        elif self.action_mode == 2:
            return np.array([self.flip_control(vel_rot_desired, obs[1], self.epsilon)]), None
        
    def flip_control(self, target, current, eps):
        action = 1
        if target-current > eps:
            action = 2
        elif target-current < -eps:
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
    
    def compute_target_vel(self, angle):
        if abs(angle) > self.angle:
            vel_rot_desired = angle/abs(angle)
        else:
            vel_rot_desired = angle/self.angle
        return vel_rot_desired