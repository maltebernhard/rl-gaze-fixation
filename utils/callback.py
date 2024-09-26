from typing import Dict
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback for plotting the training progress
class ModularAgentCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.model_name = None
        self.episode_rewards: Dict[str,list] = {}
        self.current_episode_reward = 0

    def set_model_name(self, model_name):
        self.model_name = model_name
        if model_name not in self.episode_rewards.keys():
            self.episode_rewards[model_name] = []

    def plot_training_progress(self):
        # Plot rewards for each callback
        for key, rewards in self.episode_rewards.items():
            plt.plot(rewards, label=key)
        # Add labels and title
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(f'Training Progress')
        plt.legend()
        plt.grid()
        plt.show()

    def _on_step(self) -> bool:
        # Accumulate the reward for the current episode
        self.current_episode_reward += self.locals['rewards'][0]
        #self.last_observation = self.locals['observation']

        # Check if the episode has ended
        if self.locals['dones'][0]:
            # Save the cumulative reward for this episode
            self.episode_rewards[self.model_name].append(self.current_episode_reward)
            # Print the reward for this episode
            print(f"Episode reward: {self.current_episode_reward}")
            #print(f"Last observation: {self.last_observation}")
            # Reset the cumulative reward for the next episode
            self.current_episode_reward = 0

        return True

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