from stable_baselines3.common.callbacks import BaseCallback

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