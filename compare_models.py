from typing import List
import yaml
from agent.base_agent import BaseAgent
from utils.plotting import plot_training_progress_multiple

total_timesteps = 250000


base_agents_mixture = [BaseAgent.load()]
base_agents_monolithic = [BaseAgent.load()]

plot_training_progress_multiple([agent.callback for agent in base_agents_mixture+base_agents_monolithic], savepath="mixture.png")
