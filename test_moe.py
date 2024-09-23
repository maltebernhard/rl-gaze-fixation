from typing import Dict
import yaml
import gymnasium as gym
import numpy as np
from agent.base_agent import BaseAgent
from environment.base_env import BaseEnv
from environment.gaze_fix_env import GazeFixEnv

env_config = {
    "timestep":            0.05,
    "episode_length":      30.0,
    "world_size":          50.0,
    "robot_sensor_angle":  np.pi * 2.0,
    "robot_max_vel":       8.0,
    "robot_max_vel_rot":   3.0,
    "robot_max_acc":       8.0,
    "robot_max_acc_rot":   10.0,
    "action_mode":         1,
    "observe_distance":    False,
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

with open("./config/agent_config.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

base_agent = BaseAgent(model_config, env_config)
base_agent.visualize_agent_tree("test_filename")

#base_agent.learn_agent(1, 100000)
#base_agent.save()
for episode in range(10):
    base_agent.run_agent(1, timesteps=10000, prints=True)
