from typing import List
import yaml
from agent.base_agent import BaseAgent
from utils.plotting import plot_training_progress_multiple

with open("./config/env/three_obstacles.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/(ppo_ppo)_ppo_gaze.yaml") as file:
    model_config_monolithic = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/ppo_gazefix.yaml") as file:
    model_config_modular = yaml.load(file, Loader=yaml.FullLoader)

base_agents_modular: List[BaseAgent] = []
base_agents_monolithic: List[BaseAgent] = []

for i in range(5):
    model_config_modular["random_seed"] = i
    model_config_monolithic["random_seed"] = i
    base_agents_modular.append(BaseAgent(model_config_modular, env_config))
    base_agents_monolithic.append(BaseAgent(model_config_monolithic, env_config))

    base_agents_modular[-1].learn(5000000, 2048, True)
    base_agents_monolithic[-1].learn(5000000, 2048, True)

plot_training_progress_multiple([agent.callback for agent in (base_agents_modular+base_agents_monolithic)], savepath="./compare_models")