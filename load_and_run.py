import gymnasium as gym
import yaml
from helpers import prompt_zip_file_selection
from model.model import Model

# ==============================================

filename = prompt_zip_file_selection()

with open(filename[:-5] + 'env_config.yaml', 'r') as file:
    env_config = yaml.load(file, Loader=yaml.SafeLoader)
    env_config["episode_length"] = 20.0
with open(filename[:-5] + 'model_config.yaml', 'r') as file:
    model_config = yaml.load(file, Loader=yaml.SafeLoader)

# Create and wrap the environment
env = gym.make(id='GazeFixAgent',
               config = env_config
              )

model = Model(env, model_config)

model.load(filename)

model.run_model(3,1)