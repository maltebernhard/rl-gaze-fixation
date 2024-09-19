import gymnasium as gym
from stable_baselines3 import PPO
import yaml
from helpers import prompt_zip_file_selection, user_prompt
from model.model import Model
from training_logging.plotting import plot_training_progress

# ==============================================================

filename = prompt_zip_file_selection()

with open(filename[:-5] + 'env_config.yaml', 'r') as file:
    env_config = yaml.load(file, Loader=yaml.SafeLoader)
with open(filename[:-5] + 'model_config.yaml', 'r') as file:
    model_config = yaml.load(file, Loader=yaml.SafeLoader)

env_config["use_obstacles"] = True
model_config["total_timesteps"] = 200000

env = gym.make(id = 'GazeFixAgent',
               config = env_config)

model = PPO.load(filename, env)
model = Model(env, model_config, model)
model.reset()
model.learn(total_timesteps=model_config["total_timesteps"])

plot_training_progress(model.callback)

if user_prompt("Do you want a demo run?"):
    model.run_model()

if user_prompt("Do you want to save the model?"):
    model.save()