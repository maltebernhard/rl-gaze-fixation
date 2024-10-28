from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import wandb

# ===========================================================================================

class BaseAgentCallback:
    def __init__(self, agent_name, observation_keys = [], action_keys = ["frontal_movement", "lateral_movement", "rotational_movement"]):
        self.agent_name = agent_name
        self.observation_keys = observation_keys
        self.action_keys = action_keys
        self.run = None
        self.episode_rewards: list = []
        self.reset()
        # TODO: regulate better
        self.log_stepwise = False

    def reset(self) -> None:
        self.current_episode_reward = np.zeros(5)

    def log(self, action, observation, reward, done) -> bool:
        # Accumulate the reward for the current episode
        self.current_episode_reward += reward

        # wandb logging
        if self.log_stepwise:
            log_data = {f"base_{self.get_reward_label(i)}": reward[i] for i in range(5)}
            log_data.update({f"base_{self.action_keys[i]}": action[i] for i in range(3)})
            log_data.update({f"base_{self.observation_keys[i]}": observation[i] for i in range(len(observation))})
            self.run.log(log_data)

        # Check if the episode has ended
        if done:
            # Save the cumulative reward for this episode
            self.episode_rewards.append(self.current_episode_reward)
            # Print the reward for this episode
            tqdm.write(f"Base episode rewards: {self.current_episode_reward}")
            self.run.log({f"base_episode_{self.get_reward_label(i)}": self.current_episode_reward[i] for i in range(5)})
            self.reset()

        return True
    
    def end_training(self) -> None:
        self.current_episode_reward = 0
        self.run = None
        self.reset()

    # ---------------------------------- plotting ----------------------------------

    def get_reward_label(self, i: int):
        if i == 0: return "Target_Proximity_Reward"
        elif i == 1: return "Time_Penalty"
        elif i == 2: return "Obstacle_Proximity_Penalty"
        elif i == 3: return "Energy_Waste_Penalty"
        elif i == 4: return "Collision_Penalty"

    def plot_training_progress(self, show=True, savefolder=None):
        for i in range(5):
            rewards = [episode_reward[i] for episode_reward in self.episode_rewards]
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

# ===========================================================================================

# Custom callback for plotting the training progress
class ModularAgentCallback(BaseCallback):
    def __init__(self, verbose=0, model_name = "Base", observation_keys = [], action_keys = []):
        super().__init__(verbose)
        self.agent_name = model_name
        self.observation_keys = observation_keys
        self.action_keys = action_keys
        self.run = None
        self.episode_rewards: list = []
        self.reset()

    def reset(self) -> None:
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate the reward for the current episode
        self.current_episode_reward += self.locals['rewards'][0]

        # wandb logging
        reward = self.locals['rewards'][0]
        action = self.locals['clipped_actions'][0]
        observation = self.locals['new_obs'][0]
        done = self.locals['dones'][0]

        log_data = {f"{self.agent_name}_reward": reward}
        log_data.update({f"{self.agent_name}_{self.action_keys[i]}": action[i] for i in range(len(action))})
        log_data.update({f"{self.agent_name}_{self.observation_keys[i]}": observation[i] for i in range(len(observation))})
        self.run.log(log_data)

        # Check if the episode has ended
        if done:
            # Save the cumulative reward for this episode
            self.episode_rewards.append(self.current_episode_reward)
            # Print the reward for this episode
            tqdm.write(f"{self.agent_name} episode reward: {self.current_episode_reward}")
            self.run.log({f"{self.agent_name}_episode_reward": self.current_episode_reward})
            # Reset the cumulative reward for the next episode
            self.reset()
        return True
    
    def _on_training_end(self) -> None:
        self.current_episode_reward = 0

    def plot_training_progress(self, show=True, savefolder=None):
        plt.plot(self.episode_rewards, label=f"{self.agent_name} Episode Rewards")
        # Add labels and title
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title(f'Training Progress')
        plt.grid()
        plt.legend()
        if show:
            plt.show()
        if savefolder is not None:
            plt.savefig(f"{savefolder}/training_progress_.png")
        plt.close()