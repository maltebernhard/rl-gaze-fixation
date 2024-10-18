from abc import abstractmethod
from environment.structure_env import StructureEnv
from utils.plotting import ModularAgentCallback
import wandb
# =============================================================================

class Model:
    id = "M"
    observation_keys = []

    def __init__(self, env: StructureEnv, config={}) -> None:
        self.env = env

    def learn(self, total_timesteps: int, callback: ModularAgentCallback):
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
                callback.locals['action'] = action
                callback.locals['observation'] = obs
                callback.locals['dones'] = [done]
                callback._on_step()
                timestep += 1
                total_reward += reward
                if done:
                    obs, info = self.env.reset()
            #print("Episode reward: {}".format(total_reward))

    @abstractmethod
    def predict(self, obs, deterministic = True):
        raise NotImplementedError

    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass