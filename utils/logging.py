import copy
import os
import gymnasium as gym
from datetime import datetime
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import wandb
import yaml

from agent.base_agent import BaseAgent

def create_seeded_agents(agent_config_path, num_agents, env_config_path):
    with open(env_config_path) as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open(agent_config_path) as file:
        agent_config: dict = yaml.load(file, Loader=yaml.FullLoader)
    agent_configs = [copy.deepcopy(agent_config) for _ in range(num_agents)]
    for i in range(num_agents):
        for j in range(len(agent_configs[i]["agents"])):
            if agent_configs[i]["agents"][j]["model_type"] == "PPO":
                agent_configs[i]["agents"][j]["seed"] += i
    base_agents = [BaseAgent(agent_configs[i], env_config) for i in range(num_agents)]
    return base_agents

def training_run(project_name, run_name, agent_config, env_config, training_timesteps, record_video=False):
    run = create_run(project_name=project_name, model_config=agent_config, name=run_name, group=run_name)
    agent = create_agent(agent_config=agent_config, env_config=env_config, run_id=run.id)
    agent.set_wandb_callback()
    agent.learn(total_timesteps=training_timesteps, timesteps_per_run=4096, save=True, plot=False)
    #save_artifact(run, run_name, folder)
    run.finish()

def create_agent(agent_config, env_config, run_id) -> BaseAgent:
    def make_env():
        env = gym.make(
            id='GazeFixEnv',
            config=env_config
        )
        return Monitor(env)  # record stats such as returns

    env = DummyVecEnv([make_env])
    # model = PPO(model_config["policy_type"], env, learning_rate=model_config["learning_rate"], verbose=1, seed=model_config["seed"], tensorboard_log=f"runs/{run_id}")
    # model = PPO(agent_config["policy_type"], env, learning_rate=agent_config["learning_rate"], verbose=1, seed=agent_config["seed"])
    # model = Model(env.envs[0].env.unwrapped, agent_config, model)
    # model.reset()

    agent = BaseAgent(agent_config, env_config)

    return agent

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

def train_model(agent: BaseAgent, total_timesteps):
    agent.learn(
        total_timesteps=total_timesteps,
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