from datetime import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
import environment
from wandb.integration.sb3 import WandbCallback

from model.model import Model

# ============================= config =================================

# ------------------------- env params --------------------------------

# 1 - Acceleration in range
# 2 - Acceleration -1 , 0, 1
action_mode = 2
timestep = 0.05
episode_length = 30.0
world_size = 50.0
target_distance = 10.0
reward_margin = 3.0
wall_collision = False
num_obstacles = 0

robot_sensor_angle = np.pi / 2
robot_max_vel = 8.0
robot_max_vel_rot = 3.0
robot_max_acc = 8.0
robot_max_acc_rot = 10.0

env_seed = 12345

use_contingencies = True

# ------------------------- training params ----------------------------

# 0 - Baseline
# 1 - Deep Q-Network
# 2 - PPO
# 3 - A2C
model_selection = 2
policy_type = "MlpPolicy"
learning_rate = 0.02
training_timesteps = 1000000
model_seed = 12345

# ==============================================================

set_random_seed(env_seed)

model_config = {
    "model_selection" : model_selection,
    "policy_type": policy_type,
    "learning_rate": learning_rate,
    "total_timesteps": training_timesteps,
    "env_name": "GazeFixation",
    "seed": model_seed
}

env_config = {
    "CONT" : {
        "timestep" : timestep,
        "episode_length" : episode_length,
        "world_size" : world_size,
        "robot_sensor_angle" : robot_sensor_angle,
        "robot_max_vel" : robot_max_vel,
        "robot_max_vel_rot" : robot_max_vel_rot,
        "robot_max_acc" : robot_max_acc,
        "robot_max_acc_rot" : robot_max_acc_rot,
        "action_mode" : action_mode,
        "target_distance" : target_distance,
        "target_distance_reward_margin" : reward_margin,
        "wall_collision" : wall_collision,
        "num_obstacles" : num_obstacles,
        "use_contingencies" : True,
        "seed" : env_seed
    },
    "FREE" : {
        "timestep" : timestep,
        "episode_length" : episode_length,
        "world_size" : world_size,
        "robot_sensor_angle" : robot_sensor_angle,
        "robot_max_vel" : robot_max_vel,
        "robot_max_vel_rot" : robot_max_vel_rot,
        "robot_max_acc" : robot_max_acc,
        "robot_max_acc_rot" : robot_max_acc_rot,
        "action_mode" : action_mode,
        "target_distance" : target_distance,
        "target_distance_reward_margin" : reward_margin,
        "wall_collision" : wall_collision,
        "num_obstacles" : num_obstacles,
        "use_contingencies" : False,
        "seed" : env_seed
    }
}

for run_name in ["CONT","FREE"]:
    run = wandb.init(
        project="Master Thesis",
        name=datetime.today().strftime('%Y-%m-%d_%H-%M')+f"_{run_name}",
        config=model_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,       # auto-upload the videos of agents playing the game
        save_code=True,         # optional
    )

    def make_env():
        env = gym.make(id='GazeFixAgent',
                       config=env_config[run_name]
                      )
        env.reset(seed=env_seed)
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