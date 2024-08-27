from datetime import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
import yaml
import environment
from wandb.integration.sb3 import WandbCallback

from helpers import prompt_zip_file_selection
from model.model import Model

# ==============================================================

filename = prompt_zip_file_selection()

with open(filename[:-5] + 'env_config.yaml', 'r') as file:
    env_config = yaml.load(file, Loader=yaml.SafeLoader)
with open(filename[:-5] + 'model_config.yaml', 'r') as file:
    model_config = yaml.load(file, Loader=yaml.SafeLoader)


run = wandb.init(
    project="Master Thesis",
    name=datetime.today().strftime('%Y-%m-%d_%H-%M')+f"_CONT_pretrained",
    config=model_config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,       # auto-upload the videos of agents playing the game
    save_code=True,         # optional
)

def make_env():
    env = gym.make(id='GazeFixAgent',
                    config=env_config
                    )
    env.reset(seed=env_config["seed"])
    return Monitor(env)  # record stats such as returns


env = DummyVecEnv([make_env])

# env = VecVideoRecorder(
#     env,
#     f"videos/{run.id}",
#     record_video_trigger=lambda x: x % 2000 == 0,
#     video_length=200,
# )

model = PPO.load(filename, env.envs[0].env.unwrapped)
model = Model(env.envs[0].env.unwrapped, model_config, model)
model.learn(
    total_timesteps=model_config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

model.save()

run.finish()