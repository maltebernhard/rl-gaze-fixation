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

    @abstractmethod
    def predict(self, obs, deterministic = True):
        raise NotImplementedError

    def save(self, file_path: str):
        pass
        
    @classmethod
    def load(cls, file_path: str):
        pass