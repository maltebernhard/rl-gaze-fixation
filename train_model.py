import yaml
from agent.base_agent import BaseAgent
from utils.plotting import plot_training_progress_modular

with open("./config/env/three_obstacles.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/(ppo_ppo)_ppo_gaze.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
base_agent = BaseAgent(model_config, env_config)

base_agent.learn(1000000, 2048, True)
