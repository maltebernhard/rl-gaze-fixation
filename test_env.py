import gymnasium as gym
import env
import numpy as np

# ============================================

timestep = 0.01 # timestep in seconds

# ==============================================

# Create and wrap the environment
envi = gym.make(id='GazeFixAgent',
               timestep = timestep)

# envi = gym.make(id='GazeFixEnv',
#                timestep = timestep)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

step = 0
observation = envi.reset()
total_reward = 0
done = False
while not done:
    envi.render()
    action = envi.action_space.sample()
    #action = 2

    # if step < 100:
    #     action = [1.0, 0.0]
    # else:
    #     action = [0.0, 0.0]

    rand = np.random.random()
    action = [3.0, rand]

    observation, reward, done, truncated, info = envi.step(action)
    step += 1
    total_reward += reward
    print(f'Observation: {observation} | Action: {action}')
print("Episode finished with total reward {}".format(total_reward))

envi.close()