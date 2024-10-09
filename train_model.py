import yaml
from agent.base_agent import BaseAgent

with open("./config/env/one_obstacle.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/(targ_obst)_ppo_gaze.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
base_agent = BaseAgent(model_config, env_config)

base_agent.learn(100000, 2048, save=True, plot=False)