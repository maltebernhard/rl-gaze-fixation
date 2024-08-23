import gymnasium as gym
import yaml
import environment
from helpers import prompt_zip_file_selection
from model.model import Model

# ============================================

timestep = 0.01 # timestep in seconds

# 1 - Acceleration in range
# 2 - Acceleration -1 , 0, 1
action_mode = 2

# distance to target for reward function
target_distance = 2.0
wall_collision = False
obstacles = False

use_contingencies = True

seed = 9876543

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