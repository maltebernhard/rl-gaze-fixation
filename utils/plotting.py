from typing import List
import matplotlib.pyplot as plt

from utils.callback import ModularAgentCallback, PlottingCallback
    
def plot_training_progress(callback: PlottingCallback):
    # Plot the rewards
    plt.plot(callback.episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()

def plot_training_progress_modular(callbacks: List[ModularAgentCallback]):
    # Ensure we have at least one callback to plot
    if not callbacks:
        print("No callbacks to plot.")
        return

    # Plot rewards for each callback
    colors = plt.cm.get_cmap('tab10', len(callbacks))
    for idx, callback in enumerate(callbacks):
        color = colors(idx)
        for key, rewards in callback.episode_rewards.items():
            plt.plot(rewards, label=f'{callback.model_name} - {key}', color=color)

    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Training Progress')
    plt.legend()
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