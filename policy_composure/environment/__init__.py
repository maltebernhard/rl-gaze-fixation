import gymnasium as gym

gym.envs.register(
    id='GazeFixEnv',
    entry_point='environment.env:Environment',
)

gym.envs.register(
    id='PolicyEnv',
    entry_point='environment.structure_env:PolicyEnv',
)

gym.envs.register(
    id='ContingencyEnv',
    entry_point='environment.structure_env:ContingencyEnv',
)

gym.envs.register(
    id='MixtureEnv',
    entry_point='environment.structure_env:MixtureEnv',
)