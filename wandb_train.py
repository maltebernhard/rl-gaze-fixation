from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml

from model.model import Model

# ==============================================================

with open('./config/env_config.yaml', 'r') as file:
    env_config = yaml.load(file, Loader=yaml.SafeLoader)
with open('./config/model_config.yaml', 'r') as file:
    model_config = yaml.load(file, Loader=yaml.SafeLoader)

for run_name in ["CONT","FREE"]:
    run = wandb.init(
        project="Master Thesis",
        name=datetime.today().strftime('%Y-%m-%d_%H-%M')+f"_{run_name}",
        config=model_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,       # auto-upload the videos of agents playing the game
        save_code=True,         # optional
    )

    env_config["use_contingencies"] = (run_name == "CONT")
    def make_env():
        env = gym.make(id='GazeFixAgent',
                       config=env_config[run_name]
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

    model = PPO(model_config["policy_type"], env, learning_rate=model_config["learning_rate"], verbose=1, seed=model_config["seed"], tensorboard_log=f"runs/{run.id}")
    model = Model(env.envs[0].env.unwrapped, model_config, model)
    model.reset()
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