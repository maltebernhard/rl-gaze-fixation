from typing import Dict, List
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
            plt.plot(rewards, label=f'{callback.submodel_name} - {key}', color=color)

    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Training Progress')
    plt.legend()
    plt.show()

def plot_training_progress_multiple(callbacks: List[ModularAgentCallback], savepath = None):
    # Ensure we have at least one callback to plot
    if not callbacks:
        print("No callbacks to plot.")
        return

    # Plot rewards for each callback
    callback_dict: Dict[str,List[ModularAgentCallback]] = {}
    for callback in callbacks:
        if callback.model_name not in callback_dict.keys():
            callback_dict[callback.model_name] = [callback]
        else:
            callback_dict[callback.model_name].append(callback)
    
    colors = plt.cm.get_cmap('tab10', len(callback_dict))
    for idx, (model_name, model_callbacks) in enumerate(callback_dict.items()):
        color = colors(idx)
        all_rewards = []
        for callback in model_callbacks:
            all_rewards.append(callback.episode_rewards['base'])
        
        # Calculate mean and standard deviation
        mean_rewards = [sum(x) / len(x) for x in zip(*all_rewards)]
        std_rewards = [sum((x - mean) ** 2 for x in xs) ** 0.5 / len(xs) for xs, mean in zip(zip(*all_rewards), mean_rewards)]
        
        # Plot mean with standard deviation
        episodes = range(len(mean_rewards))
        plt.plot(episodes, mean_rewards, label=model_name, color=color)
        plt.fill_between(episodes, [m - s for m, s in zip(mean_rewards, std_rewards)], 
                         [m + s for m, s in zip(mean_rewards, std_rewards)], color=color, alpha=0.2)

    # Add labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Training Progress')
    plt.legend()
    plt.grid()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath)
    plt.close()