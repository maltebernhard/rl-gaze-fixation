
import time
import gymnasium as gym
import numpy as np
from agent.agent import Contingency, MixtureOfExperts, Policy
from contingency.contingency import GazeFixation
from model.avoid_nearest_obstacle import AvoidNearestObstacleModel
from model.towards_target import TowardsTargetModel

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
    "penalty_margin":      5.0,
    "wall_collision":      False,
    "num_obstacles":       3,
    "use_obstacles":       True,
    "use_contingencies":   True,
    "seed":                140
}

# Create and wrap the environment
env = gym.make(id='GazeFixEnv',
               config = env_config
              )

obstacle_evasion = Policy(env)
obstacle_evasion.set_model(AvoidNearestObstacleModel(env, action_space_dimensionality=2))
obstacle_evasion.set_observation_space()
obstacle_evasion.set_action_space(
    gym.spaces.Box(
        low=np.array([0.0, 0.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float64
    )
)

target_following = Policy(env)
target_following.set_model(TowardsTargetModel(env, action_space_dimensionality=2))
target_following.set_observation_space()
target_following.set_action_space(
    gym.spaces.Box(
        low=np.array([0.0, 0.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float64
    )
)

gaze_fixation1 = Contingency(env, obstacle_evasion, GazeFixation(env_config["timestep"], env_config["robot_max_vel_rot"], env_config["robot_max_acc_rot"], env_config["action_mode"]))
gaze_fixation1.set_observation_space()
gaze_fixation1.set_action_space()

gaze_fixation2 = Contingency(env, target_following, GazeFixation(env_config["timestep"], env_config["robot_max_vel_rot"], env_config["robot_max_acc_rot"], env_config["action_mode"]))
gaze_fixation2.set_observation_space()
gaze_fixation2.set_action_space()

moe = MixtureOfExperts(env, [gaze_fixation1, gaze_fixation2])
#moe.set_model()
moe.set_observation_space()
moe.set_action_space()
#moe.learn(100000)

for _ in range(5):
    moe.run(prints=True, env_seed = int(time.time()))
