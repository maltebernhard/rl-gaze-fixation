import time
import gymnasium as gym
import environment
import numpy as np
from model.model import Model

# ============================= config =================================

# 1 - Acceleration in range
# 2 - Acceleration -1 , 0, 1
action_mode = 2
timestep = 0.01
episode_length = 10.0
world_size = 50.0
target_distance = 10.0
reward_margin = 3.0
wall_collision = False
num_obstacles = 5

robot_max_vel = 8.0
robot_max_vel_rot = 3.0
robot_max_acc = 8.0
robot_max_acc_rot = 10.0

env_seed = int(time.time())

use_contingencies = True

model_selection = 0

# ==============================================================

model_config = {
    "model_selection" : model_selection
}

env_config = {
    "timestep" : timestep,
    "episode_length" : episode_length,
    "world_size" : world_size,
    "robot_max_vel" : robot_max_vel,
    "robot_max_vel_rot" : robot_max_vel_rot,
    "robot_max_acc" : robot_max_acc,
    "robot_max_acc_rot" : robot_max_acc_rot,
    "action_mode" : action_mode,
    "target_distance" : target_distance,
    "target_distance_reward_margin" : reward_margin,
    "wall_collision" : wall_collision,
    "num_obstacles" : num_obstacles,
    "use_contingencies" : use_contingencies,
    "seed" : env_seed
}

# Create and wrap the environment
env = gym.make(id='GazeFixAgent',
               config = env_config
              )

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

baseline_model = Model(env, model_config)

step = 0
observation = env.reset()
total_reward = 0
done = False
while not done:
    env.render()
    #action = env.action_space.sample()

    action = baseline_model.predict(observation)[0]

    observation, reward, done, truncated, info = env.step(action)
    step += 1
    total_reward += reward
    print(f'Observation: {observation} | Action: {action}')
print("Episode finished with total reward {}".format(total_reward))

env.close()