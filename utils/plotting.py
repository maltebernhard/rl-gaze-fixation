from typing import List
import matplotlib.pyplot as plt

from utils.callback import PlottingCallback
    
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