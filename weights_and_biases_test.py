from datetime import datetime
import gymnasium as gym
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
# distance to target for reward function
target_distance = 2.0
wall_collision = False
obstacles = False
use_contingencies = True
timestep = 0.01
env_seed = 123

# ------------------------- training params ----------------------------

# 0 - Baseline
# 1 - Deep Q-Network
# 2 - PPO
# 3 - A2C
model_selection = 2
policy_type = "MlpPolicy"
learning_rate = 0.02
training_timesteps = 100000
model_seed = 45283

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
    "timestep" : timestep,
    "action_mode" : action_mode,
    "target_distance" : target_distance,
    "wall_collision" : wall_collision,
    "obstacles" : obstacles,
    "use_contingencies" : use_contingencies,
    "seed" : env_seed,
}

run = wandb.init(
    project="Master Thesis",
    name=datetime.today().strftime('%Y-%m-%d_%H-%M')+"_TEST",
    config=model_config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,       # auto-upload the videos of agents playing the game
    save_code=True,         # optional
)

def make_env():
    env = gym.make(id='GazeFixAgent',
                   config=env_config
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