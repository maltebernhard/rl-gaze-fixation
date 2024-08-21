import gymnasium as gym
import env
import numpy as np

# ============================================

timestep = 0.01 # timestep in seconds

# 1 - Acceleration in range
# 2 - Acceleration -1 , 0, 1
action_mode = 2

# distance to target for reward function
target_distance = 2.0

use_contingencies = False

# ==============================================

# Create and wrap the environment
envi = gym.make(id='GazeFixAgent',
               timestep = timestep,
               action_mode = action_mode,
               use_contingencies = use_contingencies,
               distance = target_distance
               )

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

step = 0
observation = envi.reset()
total_reward = 0
done = False
while not done:
    envi.render()
    action = envi.action_space.sample()

    # if step < 100:
    #     action = [1.0, 0.0]
    # else:
    #     action = [0.0, 0.0]

    # rand = np.random.random()
    # action = [3.0, rand]

    observation, reward, done, truncated, info = envi.step(action[0])
    step += 1
    total_reward += reward
    print(f'Observation: {observation} | Action: {action[0]}')
print("Episode finished with total reward {}".format(total_reward))

envi.close()