import time
import gymnasium as gym
import numpy as np
import yaml
from model.model import Model

# ============================= config =================================

env_config = {
    "timestep":            0.05,
    "episode_length":      5.0,
    "world_size":          50.0,
    "robot_sensor_angle":  np.pi * 2.0,
    "robot_max_vel":       8.0,
    "robot_max_vel_rot":   3.0,
    "robot_max_acc":       8.0,
    "robot_max_acc_rot":   10.0,
    "action_mode":         2,
    "observe_distance":    True,
    "target_distance":     10.0,
    "reward_margin":       10.0,
    "penalty_margin":      5.0,
    "wall_collision":      False,
    "num_obstacles":       3,
    "use_obstacles":       True,
    "use_contingencies":   True,
    "seed":                140
    #"seed":                int(time.time())
}

# with open('./config/env_config.yaml', 'r') as file:
#     env_config = yaml.load(file, Loader=yaml.SafeLoader)

# ==========================================================================

# Create and wrap the environment
env = gym.make(id='GazeFixAgent',
               config = env_config
              )

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

baseline_model = Model(env, {"model_selection":0})

baseline_model.run_model(1, 1, False)