import yaml
from agent.base_agent import BaseAgent

with open("./config/env/three_obstacles.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/(targ_obst)_ppo_gaze.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

base_agent = BaseAgent(model_config, env_config)

base_agent.learn_agent(3, 200000, plot=True)
base_agent.save()
for episode in range(10):
    base_agent.run_agent(4, timesteps=1000, prints=True)
