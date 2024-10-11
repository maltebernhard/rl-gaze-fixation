import time
import numpy as np
import yaml
from agent.base_agent import BaseAgent
import matplotlib.pyplot as plt

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

with open("./config/agent/1obst-(targ_obst_still)_mixt_gaze.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

base_agent = BaseAgent(model_config, env_config)

#base_agent.run(prints=False, steps=100000, env_seed=env_seed)

logs = []
for i in range(20):
    log = base_agent.run_agent("Mixture-Agent", timesteps=100000, prints=False, render=False, env_seed=env_seed+i)
    logs.append(log)

actions = np.array([log["actions"] for log in logs])
mean_actions = np.mean(actions, axis=0)
std_actions = np.std(actions, axis=0)

for i in range(mean_actions.shape[1]):
    if i == 0:
        label = 'Towards Target Relevance'
    elif i == 1:
        label = 'Obstacle Evasion Relevance'
    else:
        label = 'Stopping Relevance'
    plt.plot(mean_actions[:, i], label=f'{label} Mean')
    plt.fill_between(range(mean_actions.shape[0]), 
                     mean_actions[:, i] - std_actions[:, i], 
                     mean_actions[:, i] + std_actions[:, i], 
                     alpha=0.2)

plt.xlabel('Timestep')
plt.ylabel('Action Value')
plt.title('Mean and Standard Deviation of Actions over Time')
plt.legend()
plt.grid()
plt.show()