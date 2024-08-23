import gymnasium as gym
from model.model import Model
import tkinter as tk
from tkinter import messagebox
from plotting.plotting import plot_training_progress

# ==============================================================

# ------------------------- env params --------------------------------

# 1 - Acceleration in range
# 2 - Acceleration -1 , 0, 1
action_mode = 2
# distance to target for reward function
target_distance = 5.0
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
training_timesteps = 10000
model_seed = 45283

# ==============================================================

def user_prompt(question: str):
    # Create the root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # Ask the user if they want to save the model
    response = messagebox.askyesno("User Prompt", question)
    # Destroy the root window
    root.destroy()
    return response

# ==============================================================

env_config = {
    "timestep" : timestep,
    "action_mode" : action_mode,
    "target_distance" : target_distance,
    "wall_collision" : wall_collision,
    "obstacles" : obstacles,
    "use_contingencies" : use_contingencies,
    "seed" : env_seed,
}

model_config = {
    "model_selection" : model_selection,
    "policy_type": policy_type,
    "learning_rate": learning_rate,
    "total_timesteps": training_timesteps,
    "env_name": "GazeFixation",
    "seed": model_seed
}

env = gym.make(id='GazeFixAgent',
               config = env_config)

model = Model(env, model_selection)

if model_selection != 0:
    model.learn(training_timesteps)
    plot_training_progress(model.callback)

if model_selection == 0 or user_prompt("Do you want a demo run?"):
    model.run_model()

if model_selection != 0:
    if user_prompt("Do you want to save the model?"):
        model.save()