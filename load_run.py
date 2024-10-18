from agent.base_agent import BaseAgent
from utils.plotting import plot_actions_observations


agent = BaseAgent.load()

agent.run(timesteps=1000, env_seed=5, prints=True)

plot_actions_observations(agent.agents["Mixture-Agent"], num_logs=20, env_seed=5, savepath=None)