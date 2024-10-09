from typing import List
import yaml
from agent.base_agent import BaseAgent
from utils.plotting import plot_training_progress_multiple

total_timesteps = 250000

with open("./config/env/zero_obstacles.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/2024-10-07/ppo.yaml") as file:
    model_config_free = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/2024-10-07/ppo_gazefix.yaml") as file:
    model_config_contingent = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/2024-10-07/ppo_dist.yaml") as file:
    model_config_free_dist = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/2024-10-07/ppo_gazefix_dist.yaml") as file:
    model_config_contingent_dist = yaml.load(file, Loader=yaml.FullLoader)

base_agents_contingent: List[BaseAgent] = []
base_agents_free: List[BaseAgent] = []

base_agents_contingent_dist: List[BaseAgent] = []
base_agents_free_dist: List[BaseAgent] = []

for i in range(10):
    model_config_free["agents"][0]["seed"] = i
    model_config_contingent["agents"][0]["seed"] = i
    model_config_free_dist["agents"][0]["seed"] = i
    model_config_contingent_dist["agents"][0]["seed"] = i

    base_agents_contingent.append(BaseAgent(model_config_contingent, env_config))
    base_agents_free.append(BaseAgent(model_config_free, env_config))
    base_agents_contingent_dist.append(BaseAgent(model_config_contingent_dist, env_config))
    base_agents_free_dist.append(BaseAgent(model_config_free_dist, env_config))

    base_agents_contingent[-1].learn(total_timesteps=total_timesteps, timesteps_per_run=2048, save=True, plot=False)
    base_agents_free[-1].learn(total_timesteps=total_timesteps, timesteps_per_run=2048, save=True, plot=False)
    base_agents_contingent_dist[-1].learn(total_timesteps=total_timesteps, timesteps_per_run=2048, save=True, plot=False)
    base_agents_free_dist[-1].learn(total_timesteps=total_timesteps, timesteps_per_run=2048, save=True, plot=False)

plot_training_progress_multiple([agent.callback for agent in base_agents_free_dist+base_agents_contingent_dist], savepath="./compare_models_with_distance")
plot_training_progress_multiple([agent.callback for agent in base_agents_free+base_agents_contingent], savepath="./compare_models_without_distance")
plot_training_progress_multiple([agent.callback for agent in (base_agents_contingent+base_agents_free+base_agents_contingent_dist+base_agents_free_dist)], savepath="./compare_models")
