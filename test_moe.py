import yaml
from agent.base_agent import BaseAgent

with open("./config/env/example_env_config.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/example_agent_config.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

base_agent = BaseAgent(model_config, env_config)
base_agent.visualize_agent_tree("test_filename")

#base_agent.learn_agent(1, 100000)
#base_agent.save()
for episode in range(10):
    base_agent.run_agent(4, timesteps=10000, prints=True)
