import numpy as np

class Contingency:
    def __init__(self):
        pass

    def contingent_action(self, observation, action):
        raise NotImplementedError
    
class GazeFixation(Contingency):
    def __init__(self, max_acc_phi):
        super().__init__()
        self.max_acc = max_acc_phi

    def contingent_action(self, obs, act):
        d_phi_target = self.compute_target_vel(obs[0], obs[1])
        return np.concatenate([act, np.array([self.pd_control(d_phi_target, obs[1])])])
        
    def pd_control(self, x, del_x):
        K_p = 10.0
        K_d = 1.0
        return K_p * x - K_d * del_x
    
    def p_control(self, x):
        K_p = 10.0
        return K_p * x
    
    def compute_target_vel(self, angle, d_angle):
        K_p = 2.0
        # Proportional term to compute desired velocity
        desired_d_angle = K_p * angle
        # Derivative term to control acceleration to achieve desired velocity
        return desired_d_angle - d_angle