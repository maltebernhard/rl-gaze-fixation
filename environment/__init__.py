import gymnasium as gym

gym.envs.register(
    id='GazeFixEnv',
    entry_point='environment.env:GazeFixEnv',
)

gym.envs.register(
    id='StructureEnv',
    entry_point='environment.structure_env:StructureEnv',
)

gym.envs.register(
    id='BaseEnv',
    entry_point='environment.base_env:BaseEnv',
)