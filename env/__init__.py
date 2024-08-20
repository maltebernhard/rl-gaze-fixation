import gymnasium as gym

gym.envs.register(
    id='GazeFixEnv',
    entry_point='env.env:Environment',
)

gym.envs.register(
    id='GazeFixAgent',
    entry_point='env.agent:Agent',
)