from utils.plotting import plot_training_progress_multiple
import os
from agent.base_agent import BaseAgent


agents = []

directory = "./training_data/2024-10-07"

for subdir in os.listdir(directory):
    subdir_path = os.path.join(directory, subdir)
    if os.path.isdir(subdir_path):
        agent = BaseAgent.load(subdir_path+"/")
        agents.append(agent)

distance_agents = [agent.callback for agent in agents if "with distance" in agent.callback.model_name]
non_distance_agents = [agent.callback for agent in agents if "without distance" in agent.callback.model_name]

plot_training_progress_multiple(distance_agents, savepath="./compare_models_with_distance")
plot_training_progress_multiple(non_distance_agents, savepath="./compare_models_without_distance")
plot_training_progress_multiple(distance_agents+non_distance_agents, savepath="./compare_models")
