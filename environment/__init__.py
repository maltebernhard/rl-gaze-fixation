import gymnasium as gym
import numpy as np

np.set_printoptions(precision=2)

gym.envs.register(
    id='GazeFixEnv',
    entry_point='environment.gaze_fix_env:GazeFixEnv',
)

gym.envs.register(
    id='StructureEnv',
    entry_point='environment.structure_env:StructureEnv',
)

gym.envs.register(
    id='BaseEnv',
    entry_point='environment.base_env:BaseEnv',
)