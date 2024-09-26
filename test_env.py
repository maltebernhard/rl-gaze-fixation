import time
import numpy as np
import yaml
from agent.base_agent import BaseAgent
import environment

# ============================= config =================================

env_config = {
    "timestep":            0.05,
    "episode_length":      60.0,
    "world_size":          50.0,
    "robot_sensor_angle":  np.pi * 2.0,
    "robot_max_vel":       8.0,
    "robot_max_vel_rot":   3.0,
    "robot_max_acc":       8.0,
    "robot_max_acc_rot":   10.0,
    "action_mode":         1,
    "target_distance":     10.0,
    "reward_margin":       10.0,
    "penalty_margin":      5.0,
    "wall_collision":      False,
    "num_obstacles":       3,
    "use_obstacles":       True,
    "use_contingencies":   True,
    "seed":                1
    #"seed":                int(time.time())
}

with open("./config/agent/(targ_obst)_mixt_gaze.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

base_agent = BaseAgent(model_config, env_config)

base_agent.run(1000, False)