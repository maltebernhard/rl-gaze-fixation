import gymnasium as gym

gym.envs.register(
    id='GazeFixEnv',
    entry_point='environment.env:Environment',
)

gym.envs.register(
    id='GazeFixAgent',
    entry_point='environment.agent:Agent',
)

gym.envs.register(
    id='MixtureEnv',
    entry_point='environment.mixture_env:MixtureEnv',
)