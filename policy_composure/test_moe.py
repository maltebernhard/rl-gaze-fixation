
import time
import gymnasium as gym
import numpy as np
from agent.agent import BaseAgent, Contingency, MixtureOfExperts, Policy
from model.model import AvoidNearestObstacleModel, TowardsTargetModel, TargetFollowingObstacleEvasionMixtureModel, GazeFixationModel
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

def create_mixture(model_type, mixture_mode):
    env = gym.make(
        id='GazeFixEnv',
        config = env_config
    )

    base_env = gym.make(
        id='BaseEnv',
        env = env,
    )
    
    obstacle_evasion = Policy(base_env = base_env, observation_keys=[item for o in range(3) for item in [f"obstacle{o+1}_offset_angle",f"obstacle{o+1}_coverage"]])
    obstacle_evasion.set_model(AvoidNearestObstacleModel(base_env, action_space_dimensionality=2))

    target_following = Policy(base_env = base_env, observation_keys=["target_offset_angle"])
    target_following.set_model(TowardsTargetModel(base_env, action_space_dimensionality=2))

    gaze_fixation1 = Contingency(base_env = base_env, contingent_agent=obstacle_evasion, observation_keys=["target_offset_angle", "del_target_offset_angle"])
    gaze_fixation1.set_model(GazeFixationModel(env_config["timestep"], env_config["robot_max_vel_rot"], env_config["robot_max_acc_rot"], env_config["action_mode"]))

    gaze_fixation2 = Contingency(base_env = base_env, contingent_agent=target_following, observation_keys=["target_offset_angle", "del_target_offset_angle"])
    gaze_fixation2.set_model(GazeFixationModel(env_config["timestep"], env_config["robot_max_vel_rot"], env_config["robot_max_acc_rot"], env_config["action_mode"]))

    moe = MixtureOfExperts(base_env, observation_keys=[item for o in range(3) for item in [f"obstacle{o+1}_offset_angle",f"obstacle{o+1}_coverage"]], experts=[gaze_fixation1,gaze_fixation2], mixture_mode=mixture_mode)
    if model_type == 1:
        moe.set_model(TargetFollowingObstacleEvasionMixtureModel(moe.env, mixture_mode=mixture_mode, action_space_dimensionality=3))
    elif model_type == 2:
        moe.set_model(seed=140)
    else:
        raise Exception("Model type not supported.")

    base_agent = BaseAgent(agents=[obstacle_evasion, target_following, gaze_fixation1, gaze_fixation2, moe])
    base_agent.set_last_agent(moe)
    base_env.set_base_agent(base_agent)

    base_env.reset()
    for agent in [obstacle_evasion, target_following, gaze_fixation1, gaze_fixation2, moe]:
        agent.env.reset()

    return moe

model = create_mixture(model_type=1, mixture_mode=1)
model.learn(1000)
model.save("training_data/moe1/mixture_model")
#model.run(prints=True)
plot_training_progress(model.callback)

# plot_training_progress_multiple(
#     [
#         moe1.callback,
#         moe2.callback,
#         moe3.callback,
#         moe4.callback
#     ]
# )
