from typing import List
from agent.base_agent import BaseAgent
from utils.logging import create_seeded_agents

# ==============================================================================
num_agents = 10

env_path = "./config/env/one_obstacle.yaml"

#agent_path = "./config/agent/1obst-(targ_obst_still_ppo)_ppo_gaze.yaml"
#agent_path = "./config/agent/1obst-(targ_obst_still)_ppo_gaze.yaml"
agent_path1 = "./config/agent/1obst-(targ_obst_still_left_right)_ppo_gaze.yaml"
agent_path2 = "./config/agent/ppo_gazefix.yaml"
agent_path3 = "./config/agent/1obst-(targ_obst_left)_ppo_gaze.yaml"

base_agents_1: List[BaseAgent] = create_seeded_agents(agent_path3, num_agents, env_path)
base_agents_2: List[BaseAgent] = create_seeded_agents(agent_path2, num_agents, env_path)
for i in range(num_agents):
    base_agents_1[i].learn(total_timesteps=200000, timesteps_per_run=2048, save=True, plot=False)
    base_agents_2[i].learn(total_timesteps=200000, timesteps_per_run=2048, save=True, plot=False)

# base_agent = create_seeded_agents(agent_path3, 1, env_path)[0]
# base_agent.learn(total_timesteps=200000, timesteps_per_run=2048, save=True, plot=False)