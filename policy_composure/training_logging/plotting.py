from typing import List
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Custom callback for plotting the training progress
class PlottingCallback(BaseCallback):
    def __init__(self, model_name, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.model_name = model_name
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate the reward for the current episode
        self.current_episode_reward += self.locals['rewards'][0]

        # Check if the episode has ended
        if self.locals['dones'][0]:
            # Save the cumulative reward for this episode
            self.episode_rewards.append(self.current_episode_reward)
            # Print the reward for this episode
            print(f"Episode reward: {self.current_episode_reward}")
            # Reset the cumulative reward for the next episode
            self.current_episode_reward = 0

        return True
    
def plot_training_progress(callback: PlottingCallback):
    # Plot the rewards
    plt.plot(callback.episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()

def plot_training_progress_multiple(callbacks: List[PlottingCallback]):
    # Ensure we have at least one callback to plot
    if not callbacks:
        print("No callbacks to plot.")
        return

    # Plot rewards for each callback
    for callback in callbacks:
        plt.plot(callback.episode_rewards, label=callback.model_name)

    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Training Progress')
    plt.legend()
    plt.show()