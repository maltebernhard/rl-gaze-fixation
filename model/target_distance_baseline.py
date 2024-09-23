import numpy as np
from environment.agent import Agent

from training_logging.plotting import PlottingCallback

class TargetDistanceBaselineModel:
    def __init__(self, env: Agent):
        self.env = env
        self.action_mode = self.env.unwrapped.action_mode
        self.timestep = self.env.unwrapped.timestep
        self.observe_distance = self.env.unwrapped.observe_distance
        self.state = None
        self.action = None

        self.target_distance = self.env.unwrapped.config["target_distance"]

        self.max_acc = self.env.unwrapped.robot.max_acc
        
    def learn(self, total_timesteps: int, callback: PlottingCallback):
        obs, info = self.env.reset()
        timestep = 0
        while timestep < total_timesteps:
            total_reward = 0
            done = False
            while not done:
                action, _states = self.predict(obs, deterministic=True)
                # print(f'Observation: {obs} | Action: {action}')
                obs, reward, done, truncated, info = self.env.step(action)
                callback.locals['rewards'] = [reward]
                callback.locals['dones'] = [done]
                callback._on_step()
                timestep += 1
                total_reward += reward
                if done:
                    obs, info = self.env.reset()
            print("Episode reward: {}".format(total_reward))

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

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        if self.state is None:
            self.state = state
        if not self.observe_distance:
            state = self.estimate_distance(state)

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