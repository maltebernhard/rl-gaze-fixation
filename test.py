import numpy as np
from utils.plotting import plot_training_progress_multiple
import os
from agent.base_agent import BaseAgent


# agents = []

# directory = "./training_data/2024-10-09"

# for subdir in os.listdir(directory):
#     subdir_path = os.path.join(directory, subdir)
#     if os.path.isdir(subdir_path):
#         agent = BaseAgent.load(subdir_path+"/")
#         agents.append(agent)

# distance_agents = [agent.callback for agent in agents if "with distance" in agent.callback.model_name]
# non_distance_agents = [agent.callback for agent in agents if "without distance" in agent.callback.model_name]

# plot_training_progress_multiple(distance_agents, savepath="./compare_models_with_distance")
# plot_training_progress_multiple(non_distance_agents, savepath="./compare_models_without_distance")
# plot_training_progress_multiple(distance_agents+non_distance_agents, savepath="./compare_models")



base_agent = BaseAgent.load()
model_config = base_agent.agent_config
trainable_agent = model_config["agents"][0]["id"]
for agent in model_config["agents"][1:]:
    if agent["model_type"] == "PPO":
        trainable_agent = agent["id"]
        break
log = base_agent.run_agent(trainable_agent, timesteps=100000, prints=False, render=False)

import matplotlib.pyplot as plt

actions = [action[0] for action in log["actions"]]
plt.plot(actions)
plt.xlabel('Timestep')
plt.ylabel('Action Value')
plt.title('Action[0] over Time')
plt.grid()
plt.show()