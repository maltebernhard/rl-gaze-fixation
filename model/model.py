from datetime import datetime
import pygame
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.utils import set_random_seed
import yaml
from environment.agent import Agent
from model.baseline import BaselineModel
from training_logging.plotting import PlottingCallback

# =======================================================

class Model:
    def __init__(self, agent: Agent, model_config, model=None) -> None:
        self.agent = agent
        self.config = model_config
        self.model_selection = self.config["model_selection"]
        self.timesteps_learned = 0
        if model is not None:
            self.model = model
            self.set_model_name()
        else:
            self.set_model()
        # Create the callback
        self.callback = PlottingCallback(self.model_name)

    # ============================================= model stuff =================================================

    def set_model(self):
        # Define the model
        if self.model_selection == 0:
            self.model = BaselineModel(self.agent)
        elif self.model_selection == 1:
            self.model = DQN(self.config["policy_type"], self.agent, learning_rate=self.config["learning_rate"], exploration_initial_eps=1.0, exploration_fraction=0.9, exploration_final_eps=0.1, verbose=1, seed=self.config["seed"])
        elif self.model_selection == 2:
            self.model = PPO(self.config["policy_type"], self.agent, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        elif self.model_selection == 3:
            self.model = A2C(self.config["policy_type"], self.agent, learning_rate=self.config["learning_rate"], verbose=1, seed=self.config["seed"])
        # elif model_selection == 4:
        #     self.model = QLearning(self.agent)
        #     self.model_name = "TQL"
        else: raise Exception("Please select a model!")
        self.set_model_name()

    def set_model_name(self):
        if self.model_selection == 0:
            self.model_name = "BSL"
        elif self.model_selection == 1:
            self.model_name = "DQN"
        elif self.model_selection == 2:
            self.model_name = "PPO"
        elif self.model_selection == 3:
            self.model_name = "A2C"

    def learn(self, total_timesteps, callback=None):
        if callback is None:
            self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        else:
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.timesteps_learned += total_timesteps

    def predict(self, obs, deterministic = True):
        return self.model.predict(obs, deterministic)
    
    def reset(self):
        set_random_seed(self.config["seed"])
    
    def run_model(self, num_episodes = 1, print_info = 0, record_video = False, video_path = ""):
        try:
            for episode in range(num_episodes):
                total_reward = 0
                step = 0
                obs, info = self.agent.reset(record_video=record_video, video_path=video_path)
                done = False
                while not done:
                    action, _states = self.predict(obs)
                    if print_info != 0 and step % print_info == 0:
                        print(f'-------------------- Step {step} ----------------------')
                        print(f'Observation: {obs}')
                        print(f'Action:      {action}')
                    obs, reward, done, truncated, info = self.agent.step(action)

                    # if step == 36:
                    #     pygame.image.save(self.agent.unwrapped.env.viewer, "Test.png")

                    if print_info != 0 and step % print_info == 0:
                        print(f'Reward:      {reward}')
                    total_reward += reward
                    step += 1
                    self.agent.render()
                    if done:
                        obs, info = self.agent.reset()
                print(f"Episode {episode} finished with total reward {total_reward}")
        except KeyboardInterrupt:
            pass

    def save(self, folder = None):
        if folder is None:
            folder = "./training_data/" + datetime.today().strftime('%Y-%m-%d_%H-%M') + "/"
        config = self.agent.get_wrapper_attr('config')
        filename = "model"
        self.model.save(folder + filename)
        with open(folder + 'env_config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        with open(folder + 'model_config.yaml', 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def load(self, filename):
        # Load the trained agent
        if self.model_name == "DQN": self.model = DQN.load(filename, self.agent)
        elif self.model_name == "PPO": self.model = PPO.load(filename, self.agent)
        elif self.model_name == "A2C": self.model = A2C.load(filename, self.agent)
        # elif self.model_name == "TQL": self.model = QLearning.load(filename, self.agent)