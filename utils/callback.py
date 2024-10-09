from typing import Dict
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

# Custom callback for plotting the training progress
class ModularAgentCallback(BaseCallback):
    def __init__(self, verbose=0, model_name = ""):
        super().__init__(verbose)
        self.model_name = model_name
        self.submodel_name = None
        self.episode_rewards: Dict[str,list] = {"base": []}
        self.current_episode_reward = 0
        self.current_base_episode_reward = 0

    def set_submodel_name(self, submodel_name):
        self.submodel_name = submodel_name
        if submodel_name not in self.episode_rewards.keys():
            self.episode_rewards[submodel_name] = []

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
        plt.clf()

    def plot_training_progress(self, show=True, savefolder=None):
        if "base" in self.episode_rewards.keys():
            plt.plot(self.episode_rewards["base"], label="base")
            # Add labels and title
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.title(f'Training Progress')
            plt.grid()
            if show:
                plt.show()
            if savefolder is not None:
                plt.savefig(f"{savefolder}/training_progress_total.png")
            plt.clf()

    def _on_step(self) -> bool:
        # Accumulate the reward for the current episode
        self.current_episode_reward += self.locals['rewards'][0]
        #self.last_observation = self.locals['observation']

        # Check if the episode has ended
        if self.locals['dones'][0]:
            # Save the cumulative reward for this episode
            self.episode_rewards[self.submodel_name].append(self.current_episode_reward)
            self.episode_rewards["base"].append(self.current_base_episode_reward)
            # Print the reward for this episode
            tqdm.write(f"Episode reward: {self.current_episode_reward}")
            #print(f"Last observation: {self.last_observation}")
            # Reset the cumulative reward for the next episode
            self.current_episode_reward = 0
            self.current_base_episode_reward = 0

        return True
    
    def _on_training_end(self) -> None:
        self.current_episode_reward = 0
        self.current_base_episode_reward = 0

# Custom callback for plotting the training progress
class PlottingCallback(BaseCallback):
    def __init__(self, model_name, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate the reward for the current episode
        self.current_episode_reward += self.locals['rewards'][0]
        #self.last_observation = self.locals['observation']

        # Check if the episode has ended
        if self.locals['dones'][0]:
            # Save the cumulative reward for this episode
            self.episode_rewards.append(self.current_episode_reward)
            # Print the reward for this episode
            print(f"Episode reward: {self.current_episode_reward}")
            #print(f"Last observation: {self.last_observation}")
            # Reset the cumulative reward for the next episode
            self.current_episode_reward = 0

        return True