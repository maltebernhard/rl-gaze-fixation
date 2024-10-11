import numpy as np
import yaml
from agent.base_agent import BaseAgent
import matplotlib.pyplot as plt

with open("./config/env/one_obstacle.yaml") as file:
    env_config = yaml.load(file, Loader=yaml.FullLoader)
with open("./config/agent/1obst-(targ_obst_still_left_right)_ppo_gaze.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
base_agent = BaseAgent(model_config, env_config)

base_agent.learn(100000, 2048, save=True, plot=False)



env_seed = 5
logs = []
max_len = 0
num_logs = 20
for i in range(num_logs):
    log = base_agent.run_agent("Mixture-Agent", timesteps=100000, prints=False, render=False, env_seed=env_seed+i)
    if len(log["actions"]) > max_len:
        max_len = len(log["actions"])
    logs.append(log)

logs = [log for log in logs if len(log["actions"]) == max_len]
print(f"Hit the obstacle {num_logs - len(logs)} times.")

actions = np.array([log["actions"] for log in logs])
mean_actions = np.mean(actions, axis=0)
std_actions = np.std(actions, axis=0)

for i in range(mean_actions.shape[1]):
    if i == 0:
        label = 'Towards Target Relevance'
    elif i == 1:
        label = 'Obstacle Evasion Relevance'
    else:
        label = 'Stopping Relevance'
    plt.plot(mean_actions[:, i], label=f'{label} Mean')
    plt.fill_between(range(mean_actions.shape[0]), 
                     mean_actions[:, i] - std_actions[:, i], 
                     mean_actions[:, i] + std_actions[:, i], 
                     alpha=0.2)

plt.xlabel('Timestep')
plt.ylabel('Action Value')
plt.title('Mean and Standard Deviation of Actions over Time')
plt.legend()
plt.grid()
plt.show()