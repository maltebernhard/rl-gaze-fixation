from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml

from model.model import Model

# ==============================================================

for run_type in ["CONT","FREE"]:
    with open('./config/env_config.yaml', 'r') as file:
        env_config = yaml.load(file, Loader=yaml.SafeLoader)
    with open('./config/model_config.yaml', 'r') as file:
        model_config = yaml.load(file, Loader=yaml.SafeLoader)
    
    for num_run in range(10):
        model_config["seed"] += 1
        env_config["seed"] += 1
        run_name = datetime.today().strftime('%Y-%m-%d_%H-%M')+f"_{run_type}"
        run = wandb.init(
            project="Sandbox",
            name=run_name,
            config=model_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,      # auto-upload the videos of agents playing the game
            save_code=False,        # optional
            tags=[run_type],
        )

        env_config["use_contingencies"] = (run_type == "CONT")
        def make_env():
            env = gym.make(id='GazeFixAgent',
                        config=env_config
                        )
            env.reset(seed=env_config["seed"])
            return Monitor(env)  # record stats such as returns

        env = DummyVecEnv([make_env])

        model = PPO(model_config["policy_type"], env, learning_rate=model_config["learning_rate"], verbose=1, seed=model_config["seed"], tensorboard_log=f"runs/{run.id}")
        model = Model(env.envs[0].env.unwrapped, model_config, model)
        model.reset()
        model.learn(
            total_timesteps=model_config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                #model_save_path=f"models/{run.id}",
                model_save_path=None,
                verbose=2,
            ),
        )

        model.save(folder = "./training_data/" + run_name + "/")

        model.run_model(1, 0, True, "./training_data/" + run_name + "/")

        artifact = wandb.Artifact(name = f"{run_name}_model", type = "model")
        artifact.add_file(local_path = "./training_data/" + run_name + "/" + "PPO_2.zip", name = "PPO_2.zip")
        artifact.add_file(local_path = "./training_data/" + run_name + "/" + "env_config.yaml", name = "env_config.yaml")
        artifact.add_file(local_path = "./training_data/" + run_name + "/" + "model_config.yaml", name = "model_config.yaml")
        artifact.add_file(local_path = "./training_data/" + run_name + "/" + "GazeFixation.mp4", name = "GazeFixation.mp4")
        artifact.save()
        run.log_artifact(artifact)

        run.finish()