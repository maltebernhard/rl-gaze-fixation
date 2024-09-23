import os
import gymnasium as gym
from datetime import datetime
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import wandb
from agent.OLD_model import Model

def training_run(project_name, run_name, model_config, env_config, record_video=False):
    run = create_run(project_name=project_name, model_config=model_config, name=run_name, group=run_name)
    model = create_model(model_config=model_config, env_config=env_config, run_id=run.id)
    train_model(model, model_config["total_timesteps"])
    folder = f"./training_data/{datetime.today().strftime('%Y-%m-%d_%H-%M')}_{run_name}/"
    model.save(folder = folder)
    if record_video:
        model.run_model(1, 0, True, folder)
    save_artifact(run, run_name, folder)
    run.finish()

def create_model(model_config, env_config, run_id) -> Model:
    def make_env():
        env = gym.make(
            id='GazeFixAgent',
            config=env_config
        )
        env.reset(seed=env_config["seed"])
        return Monitor(env)  # record stats such as returns

    env = DummyVecEnv([make_env])
    #model = PPO(model_config["policy_type"], env, learning_rate=model_config["learning_rate"], verbose=1, seed=model_config["seed"], tensorboard_log=f"runs/{run_id}")
    model = PPO(model_config["policy_type"], env, learning_rate=model_config["learning_rate"], verbose=1, seed=model_config["seed"])
    model = Model(env.envs[0].env.unwrapped, model_config, model)
    model.reset()
    return model

def create_run(project_name, model_config, name = datetime.today().strftime('%Y-%m-%d_%H-%M'), tags=[], group = ""):
    run = wandb.init(
        project=project_name,
        name=name,
        config=model_config,
        group=group,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,      # auto-upload the videos of agents playing the game
        save_code=False,        # optional
        tags=tags,
    )
    return run

def train_model(model: Model, total_timesteps):
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=None,
            verbose=2,
        ),
    )

def save_artifact(run, run_name, folder):
    artifact = wandb.Artifact(name = f"{run_name}_model", type = "model")
    artifact.add_file(local_path = folder + "PPO_2.zip", name = "PPO_2.zip")
    artifact.add_file(local_path = folder + "env_config.yaml", name = "env_config.yaml")
    artifact.add_file(local_path = folder + "model_config.yaml", name = "model_config.yaml")
    if os.path.isfile(folder + "GazeFixation.mp4"):
        artifact.add_file(local_path = folder + "GazeFixation.mp4", name = "GazeFixation.mp4")
    run.log_artifact(artifact)

def download_artifact(project_name = "Sandbox", artifact_name=""):
    api = wandb.Api()
    artifact = api.artifact("rbo-malte/" + project_name + "/" + artifact_name + "_model" + ":latest")
    artifact.download("./training_data/" + artifact_name + "/")