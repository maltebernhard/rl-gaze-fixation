import time
import numpy as np
import yaml
from agent.base_agent import BaseAgent
import matplotlib.pyplot as plt

from utils.plotting import plot_actions_observations

# ============================= config =================================

env_config = {
    "timestep":            0.01,
    "episode_duration":    60.0,
    "world_size":          50.0,
    "robot_sensor_angle":  np.pi * 2.0,
    "robot_max_vel":       8.0,
    "robot_max_vel_rot":   3.0,
    "robot_max_acc":       8.0,
    "robot_max_acc_rot":   10.0,
    "action_mode":         3,
    "target_distance":     10.0,
    "reward_margin":       1000.0,
    "penalty_margin":      5.0,
    "wall_collision":      False,
    "num_obstacles":       3,
    "use_obstacles":       True,
}

env_seed = 5
#env_seed = int(time.time())

with open("./config/env/one_obstacle.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)

with open("./config/agent/TEST_(targ_obst_left_gaze)_mixt.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

base_agent = BaseAgent(model_config, env_config)

base_agent.visualize_action_field()

base_agent.run(prints=False, timesteps=100000, env_seed=env_seed)

#plot_actions_observations(base_agent.last_agent, 20, env_seed=env_seed)