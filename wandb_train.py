import yaml
from utils.wandb import training_run

# ==============================================================

wandb_project_name = "TestProject"

for run_name in ["CONT","FREE"]:
    with open('./config/env_config.yaml', 'r') as file:
        env_config = yaml.load(file, Loader=yaml.SafeLoader)
    with open('./config/model_config.yaml', 'r') as file:
        model_config = yaml.load(file, Loader=yaml.SafeLoader)
    
    for num_run in range(10):
        model_config["seed"] += 1
        env_config["seed"] += 1

        training_run(
            project_name = wandb_project_name,
            run_name = run_name,
            model_config = model_config,
            env_config = env_config
        )