import gymnasium as gym
import yaml
import environment
from helpers import prompt_zip_file_selection
from model.model import Model

# ==============================================

filename = prompt_zip_file_selection()

with open(filename[:-5] + 'config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# Create and wrap the environment
env = gym.make(id='GazeFixAgent',
               config = config
              )

model = Model(env, int(filename.split("_")[-1]))

model.load(filename)

model.run_model()