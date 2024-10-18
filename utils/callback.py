from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import wandb

# ===========================================================================================

# Custom callback for plotting the training progress
class ModularAgentCallback(BaseCallback):
    def __init__(self, verbose=0, model_name = ""):
        super().__init__(verbose)
        self.model_name = model_name
        self.submodel_name = None

        # TODO:
        # self.run = wandb.init(
        #     project=project_name,
        #     name=name,
        #     config=model_config,
        #     group=group,
        #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        #     monitor_gym=False,      # auto-upload the videos of agents playing the game
        #     save_code=False,        # optional
        #     tags=tags,
        # )

        self.episode_rewards: Dict[str,list] = {"base": []}
        self.current_episode_reward = 0
        self.current_base_episode_rewards = np.zeros(5)

    def reset(self) -> None:
        self.current_episode_reward = 0
        self.current_base_episode_rewards = np.zeros(5)
        self.episode_rewards = {"base": []}

    def set_submodel_name(self, submodel_name):
        self.submodel_name = submodel_name
        if submodel_name not in self.episode_rewards.keys():
            self.episode_rewards[submodel_name] = []

    def _on_step(self) -> bool:
        # Accumulate the reward for the current episode
        self.current_episode_reward += self.locals['rewards'][0]
        #self.last_observation = self.locals['observation']

        # Check if the episode has ended
        if self.locals['dones'][0]:
            # Save the cumulative reward for this episode
            self.episode_rewards[self.submodel_name].append(self.current_episode_reward)
            self.episode_rewards["base"].append(self.current_base_episode_rewards)
            # Print the reward for this episode
            tqdm.write(f"Episode reward: {self.current_base_episode_rewards}")
            #print(f"Last observation: {self.last_observation}")
            # Reset the cumulative reward for the next episode
            self.current_episode_reward = 0
            self.current_base_episode_rewards = np.zeros(5)

        #TODO: self.run.log({"reward": reward, "action": action, "observation": obs, "done": done})
        
        return True
    
    def _on_training_end(self) -> None:
        self.current_episode_reward = 0
        self.current_base_episode_rewards = 0
        #TODO: self.run.finish()

    # ---------------------------------- plotting ----------------------------------

    def plot_subagent_training_progress(self, show=True, savefolder=None):
        # Plot rewards for each callback
        for key, rewards in self.episode_rewards.items():
            if key != "base":
                plt.plot(rewards, label=key)
        # Add labels and title
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(f'Training Progress for Sub-Agents')
        plt.legend()
        plt.grid()
        if show:
            plt.show()
        if savefolder is not None:
            plt.savefig(f"{savefolder}/training_progress_sub-agents.png")
        plt.close()

    def get_reward_label(self, i: int):
        if i == 0: return "Target Proximity Reward"
        elif i == 1: return "Time Penalty"
        elif i == 2: return "Obstacle Proximity Penalty"
        elif i == 3: return "Energy Waste Penalty"
        elif i == 4: return "Collision Penalty"

    def plot_training_progress(self, show=True, savefolder=None):
        if "base" in self.episode_rewards.keys():
            for i in range(5):
                rewards = [episode_reward[i] for episode_reward in self.episode_rewards["base"]]
                plt.plot(rewards, label=self.get_reward_label(i))
            # Add labels and title
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.title(f'Training Progress')
            plt.grid()
            plt.legend()
            if show:
                plt.show()
            if savefolder is not None:
                plt.savefig(f"{savefolder}/training_progress_total.png")
            plt.close()