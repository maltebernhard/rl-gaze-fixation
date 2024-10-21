from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from utils.callback import BaseAgentCallback, ModularAgentCallback

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

def plot_subagent_training_progress(callbacks: List[ModularAgentCallback], show=True, savefolder=None):
    # Plot rewards for each callback
    for callback in callbacks:
        plt.plot(callback.episode_rewards, label=callback.agent_name)
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

def plot_training_progress_multiple(callbacks: List[BaseAgentCallback], savepath = None):
    # Ensure we have at least one callback to plot
    if not callbacks:
        print("No callbacks to plot.")
        return

    # Plot rewards for each callback
    callback_dict: Dict[str,List[BaseAgentCallback]] = {}
    for callback in callbacks:
        if callback.agent_name in callback_dict:
            callback_dict[callback.agent_name].append(callback)
        else:
            callback_dict[callback.agent_name] = [callback]
    
    colors = plt.cm.get_cmap('tab10', len(callback_dict))
    for idx, (model_name, model_callbacks) in enumerate(callback_dict.items()):
        color = colors(idx)
        all_rewards = []
        for callback in model_callbacks:
            all_rewards.append(np.sum(np.array(callback.episode_rewards), axis=1))
        
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

def log_runs(agent, num_logs: int = 20, env_seed: int = 5):
    logs = []
    for i in range(num_logs):
        log = agent.run(timesteps=100000, env_seed=env_seed+i, render=False, prints=False)
        logs.append(log)
    return logs

def plot_actions_observations(agent, num_logs: int = 20, env_seed: int = 5, savepath = None):
    logs = []
    max_len_actions = 0
    max_len_observations = 0

    for i in range(num_logs):
        log = agent.run(timesteps=100000, env_seed=env_seed+i, render=False, prints=False)
        if len(log["actions"]) > max_len_actions:
            max_len_actions = len(log["actions"])
        if len(log["observations"]) > max_len_observations:
            max_len_observations = len(log["observations"])
        logs.append(log)

    logs = [log for log in logs if len(log["actions"]) == max_len_actions and len(log["observations"]) == max_len_observations]
    print(f"Hit the obstacle {num_logs - len(logs)} times.")

    actions = np.array([log["actions"] for log in logs])
    mean_actions = np.mean(actions, axis=0)
    std_actions = np.std(actions, axis=0)

    observations = np.array([log["observations"] for log in logs])
    mean_observations = np.mean(observations, axis=0)
    std_observations = np.std(observations, axis=0)
    observation_keys = agent.observation_keys

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 10))

    fig.canvas.manager.set_window_title(f'Agent {agent.id} Actions and Observations')

    # Plot actions
    for i in range(mean_actions.shape[1]):
        if i == 0:
            label = 'Towards Target Relevance'
        elif i == 1:
            label = 'Obstacle Evasion Relevance'
        elif i == 2:
            label = 'Stopping Relevance'
        elif i == 3:
            label = 'Left Relevance'
        elif i == 4:
            label = 'Right Relevance'
        axs[0].plot(mean_actions[:, i], label=f'{label}')
        axs[0].fill_between(range(mean_actions.shape[0]), 
                            mean_actions[:, i] - std_actions[:, i], 
                            mean_actions[:, i] + std_actions[:, i], 
                            alpha=0.2)
    axs[0].set_xlabel('Timestep')
    axs[0].set_ylabel('Action Value')
    axs[0].set_title('Mean and Standard Deviation of Actions over Time')
    axs[0].legend()
    axs[0].grid()

    # Plot observations
    for i in range(mean_observations.shape[1]):
        axs[1].plot(mean_observations[:, i], label=f'{observation_keys[i]}')
        axs[1].fill_between(range(mean_observations.shape[0]), 
                            mean_observations[:, i] - std_observations[:, i], 
                            mean_observations[:, i] + std_observations[:, i], 
                            alpha=0.2)
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('Observation Value')
    axs[1].set_title('Mean and Standard Deviation of Observations over Time')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(f"{savepath}/{agent.id}_actions_observations.png")
    plt.close()