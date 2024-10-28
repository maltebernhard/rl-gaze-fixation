from agent.base_agent import BaseAgent
from utils.plotting import plot_actions_observations


agent = BaseAgent.load()

agent.visualize_action_field()

#agent.run_agent("Mixture", timesteps=1000, env_seed=12, prints=True)