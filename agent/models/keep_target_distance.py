import numpy as np
from agent.models.model import Model
from environment.structure_env import StructureEnv

class KeepTargetDistanceModel(Model):
    observation_keys = ["target_offset_angle", "del_target_offset_angle", "vel_frontal", "vel_lateral", "vel_rot"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.action_mode = self.env.unwrapped.action_mode
        self.timestep = self.env.unwrapped.timestep
        self.state = None
        self.action = None

        self.target_distance = self.env.unwrapped.config["target_distance"]

        self.max_acc = self.env.unwrapped.robot.max_acc

    def estimate_distance(self, state):
        beta = abs(state[1]*self.timestep + state[0]-self.state[0])
        b = np.sqrt((state[2]*self.timestep)**2 + (state[3]*self.timestep)**2)

        vel = np.array([state[2], state[3]])
        del_orientation = self.state[1]*self.timestep
        vel = np.array([[np.cos(-del_orientation), -np.sin(-del_orientation)], [np.sin(-del_orientation), np.cos(-del_orientation)]]) @ vel
        alpha =  abs(np.arctan2(vel[1]*self.timestep, vel[0]*self.timestep))

        gamma = np.pi - alpha - beta
        estimated_distance = b * np.sin(gamma) / np.sin(beta) - self.target_distance
        #print(f"alpha: {alpha}\nb: {b}\nbeta: {beta}\nEstimate: {estimated_distance}")
        return np.concatenate([state,np.array([estimated_distance])])

    # TODO: reprogram to fix
    def predict(self, state, eps = 0.01, deterministic: bool = True):
        if self.state is None:
            self.state = state
            
        #state = self.estimate_distance(state)

        self.state = state

        acc_lateral = 1

        if self.state[5] > eps: acc_frontal = 2
        elif self.state[5] < 0:
            acc_frontal = 1
            if self.state[3] > 0:
                acc_lateral = 0
        else:
            acc_frontal = 1
            if self.state[3] > 0:
                acc_lateral = 0
            elif self.state[3] < 0:
                acc_frontal = 2

        self.action = np.array([2, acc_lateral, 1])
        if self.action_mode == 1:
            self.action = (self.action - np.ones(self.action.shape)) * self.max_acc
        return self.action, []
        
    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass