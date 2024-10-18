import yaml
from agent.base_agent import BaseAgent

#agent_path = "./config/agent/1obst-(targ_obst_still_ppo)_ppo_gaze.yaml"
#agent_path = "./config/agent/1obst-(targ_obst_still)_ppo_gaze.yaml"
agent_path = "./config/agent/1obst-(targ_obst_still_left_right)_ppo_gaze.yaml"

with open("./config/env/one_obstacle.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open(agent_path) as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
base_agent = BaseAgent(model_config, env_config)

base_agent.learn(200000, 2048, save=True, plot=False)

#base_agent.learn(4096, 2048, save=False, plot=True)