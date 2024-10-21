import yaml
from agent.base_agent import BaseAgent

#agent_path = "./config/agent/1obst-(targ_obst_still_ppo)_ppo_gaze.yaml"
#agent_path = "./config/agent/1obst-(targ_obst_still)_ppo_gaze.yaml"
#agent_path = "./config/agent/1obst-(targ_obst_still_left_right)_ppo_gaze.yaml"
#agent_path = "./config/agent/ppo_gazefix.yaml"

with open("./config/env/one_obstacle.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/1obst-(targ_obst_still_left_right)_ppo_gaze.yaml") as file:
    model_config1: dict = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/ppo_gazefix.yaml") as file:
    model_config2: dict = yaml.load(file, Loader=yaml.FullLoader)

model_configs1 = [model_config1.copy() for _ in range(10)]
model_configs2 = [model_config2.copy() for _ in range(10)]

for i in range(10):
    for j in range(10):
        if model_configs1[i]["agents"][j]["model_type"] == "PPO":
            model_configs1[i]["agents"][j]["seed"] = i
        if model_configs1[j]["agents"][j]["model_type"] == "PPO":
            model_configs1[j]["agents"][j]["seed"] = i

base_agents_1 = [BaseAgent(model_configs1[i], env_config) for i in range(10)]
base_agents_2 = [BaseAgent(model_configs2[i], env_config) for i in range(10)]

for i in range(10):
    base_agents_1[i].learn(200000, 2048, save=True, plot=False)
    base_agents_2[i].learn(200000, 2048, save=True, plot=False)