import numpy as np

class SMC:
    def __init__(self):
        pass

    def contingent_action(self, observation, action):
        raise NotImplementedError
    
class GazeFixation(SMC):
    def __init__(self, timestep, max_vel_rot, max_acc_rot, action_mode):
        super().__init__()
        self.max_vel = max_vel_rot
        self.max_acc = max_acc_rot
        self.angle = (self.max_vel ** 2) / (2 * self.max_acc)
        self.action_mode = action_mode
        self.epsilon = timestep * 0.5

    def contingent_action(self, obs, act):
        vel_rot_desired = self.compute_target_vel(obs[1])
        if self.action_mode == 1:
            return np.concatenate([act, np.array([self.pd_control(vel_rot_desired-obs[2], obs[2])])])
        elif self.action_mode == 2:
            return np.concatenate([act, np.array([self.flip_control(vel_rot_desired, obs[2], self.epsilon)])])
        
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