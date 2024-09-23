
from typing import Dict
import gymnasium as gym
import numpy as np
import yaml
from agent.agent import BaseAgent, StructureAgent, Contingency, MixtureOfExperts, Policy
from model.model import AvoidNearestObstacleModel, Model, TowardsTargetModel, TargetFollowingObstacleEvasionMixtureModel, GazeFixationModel
from environment.base_env import BaseEnv
from environment.env import GazeFixEnv
from training_logging.plotting import plot_training_progress, plot_training_progress_multiple

env_config = {
    "timestep":            0.05,
    "episode_length":      10.0,
    "world_size":          50.0,
    "robot_sensor_angle":  np.pi * 2.0,
    "robot_max_vel":       8.0,
    "robot_max_vel_rot":   3.0,
    "robot_max_acc":       18.0,
    "robot_max_acc_rot":   10.0,
    "action_mode":         1,
    "observe_distance":    False,
    "target_distance":     0.0,
    "reward_margin":       10.0,
    "penalty_margin":      3.0,
    "wall_collision":      False,
    "num_obstacles":       3,
    "use_obstacles":       True,
    "use_contingencies":   True,
    "seed":                140
}

def parse_models(env_config, model_config: Dict[int, dict]):
    env: GazeFixEnv = gym.make(
        id='GazeFixEnv',
        config = env_config
    )
    base_env: BaseEnv = gym.make(
        id='BaseEnv',
        env = env,
    )
    models: dict[int, StructureAgent] = {}
    for key, model_config in model_config.items():
        if model_config["type"] == "PLCY":
            models[key] = Policy(
                base_env = base_env,
                agent_config = model_config
            )
        elif model_config["type"] == "CONT":
            models[key] = Contingency(
                base_env = base_env,
                agent_config = model_config,
                contingent_agent = models[model_config["contingent_agent"]]
            )
        elif model_config["type"] == "MXTR":
            models[key] = MixtureOfExperts(
                base_env = base_env,
                agent_config = model_config,
                experts = [models[expert] for expert in model_config["experts"]]
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")

    base_agent = BaseAgent(agents=[agent for agent in models.values()])
    key = max(models.keys())
    base_agent.set_last_agent(models[key])
    base_env.set_base_agent(base_agent)

    base_env.reset()
    for agent in models.values():
        agent.env.reset()

    return models, base_env, base_agent

with open("./policy_composure/config/structure_config.yaml") as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)

models, base_env, base_agent = parse_models(env_config, model_config)

models[5].learn(total_timesteps=10000)
plot_training_progress(models[5].callback)
models[5].run()

# plot_training_progress_multiple(
#     [
#         moe1.callback,
#         moe2.callback,
#         moe3.callback,
#         moe4.callback
#     ]
# )
