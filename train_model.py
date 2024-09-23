import gymnasium as gym
import yaml
import environment
from utils.user_interface import user_prompt
from agent.agent import Policy
from utils.plotting import plot_training_progress

# ==============================================================

with open('./config/env_config.yaml', 'r') as file:
    env_config = yaml.load(file, Loader=yaml.SafeLoader)
with open('./config/OLD_model_config.yaml', 'r') as file:
    model_config = yaml.load(file, Loader=yaml.SafeLoader)
    model_selection = model_config["model_selection"]

env = gym.make(
    id = 'GazeFixEnv',
    config = env_config
)

# TODO: adjust loading, resetting and all that
model = Policy(env, model_config)
model.reset()

if model_selection != 0:
    model.learn(model_config["total_timesteps"])
    plot_training_progress(model.callback)

if model_selection == 0 or user_prompt("Do you want a demo run?"):
    model.run_model()

if model_selection != 0:
    if user_prompt("Do you want to save the model?"):
        model.save()