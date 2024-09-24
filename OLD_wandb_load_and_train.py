import yaml
from utils.user_interface import prompt_zip_file_selection
from utils.wandb import training_run

# ==============================================================

wandb_project_name = "Test_Project"
filename = prompt_zip_file_selection()

with open(filename[:-5] + 'env_config.yaml', 'r') as file:
    env_config = yaml.load(file, Loader=yaml.SafeLoader)
with open(filename[:-5] + 'model_config.yaml', 'r') as file:
    model_config = yaml.load(file, Loader=yaml.SafeLoader)

training_run(
    project_name = wandb_project_name,
    run_name = filename.split("/")[-2].split("_")[-1],
    model_config = model_config,
    env_config = env_config
)