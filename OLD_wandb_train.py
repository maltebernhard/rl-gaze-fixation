import yaml
from utils.wandb import training_run

# ==============================================================

wandb_project_name = "TestProject"

for run_name in ["CONT","FREE"]:
    with open('./config/env/zero_obstacles.yaml', 'r') as file:
        env_config = yaml.load(file, Loader=yaml.SafeLoader)
    with open('./config/agent/ppo_gazefix.yaml', 'r') as file:
        agent_config = yaml.load(file, Loader=yaml.SafeLoader)
    
    for num_run in range(2):
        # TODO: change seed in model config

        training_run(
            project_name = wandb_project_name,
            run_name = run_name,
            agent_config = agent_config,
            env_config = env_config,
            training_timesteps=10000,
        )