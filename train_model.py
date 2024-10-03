import yaml
from agent.base_agent import BaseAgent

with open("./config/env/zero_obstacles.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/ppo_gazefix.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
base_agent = BaseAgent(model_config, env_config)

base_agent.learn(100000, 2048, save=True, plot=False)