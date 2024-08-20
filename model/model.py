from stable_baselines3 import A2C, DQN, PPO
from env.agent import Agent
from model.baseline import BaselineModel
#from model.q_learning import QLearning
from plotting.plotting import PlottingCallback

# =======================================================

class Model:
    def __init__(self, agent: Agent, model_selection, model_seed = 0) -> None:
        self.agent = agent
        self.timesteps_learned = 0
        self.set_model(model_selection, model_seed)
        # Create the callback
        self.callback = PlottingCallback(self.model_name)

    # ============================================= model stuff =================================================

    def set_model(self, model_selection, model_seed):
        # Define the model
        if model_selection == 1:
            self.model = DQN('MlpPolicy', self.agent, learning_rate=0.001, exploration_initial_eps=1.0, exploration_fraction=0.9, exploration_final_eps=0.1, verbose=1, seed=model_seed)
            self.model_name = "DQN"
        elif model_selection == 2:
            self.model = PPO('MlpPolicy', self.agent, learning_rate=0.001, verbose=1, seed=model_seed)
            self.model_name = "PPO"
        elif model_selection == 3:
            self.model = A2C('MlpPolicy', self.agent, learning_rate=0.001, verbose=1, seed=model_seed)
            self.model_name = "A2C"
        elif model_selection == 0:
            self.model = BaselineModel(self.agent)
            self.model_name = "BSL"
        # elif model_selection == 4:
        #     self.model = QLearning(self.agent)
        #     self.model_name = "TQL"
        else: raise Exception("Please select a model!")

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps, callback=self.callback)
        self.timesteps_learned += total_timesteps

    def predict(self, obs, deterministic = True):
        return self.model.predict(obs, deterministic)
    
    def run_model(self):
        try:
            # Run the environment with the trained agent
            obs, info = self.agent.reset()
            dones = False
            while not dones:
                action, _states = self.predict(obs)
                # print(f'Observation: {obs} | Action: {action}')
                obs, rewards, dones, truncated, info = self.agent.step(action)
                self.agent.render()
                if dones:
                    obs, info = self.agent.reset()
        except KeyboardInterrupt:
            pass

    def save(self, folder = ""):
        filename = self.model_name + '_' + str(self.agent.env_attr('max_balls')) + '_' + str(self.agent.get_wrapper_attr('timestep')).split('.')[1] + '_' + str(self.timesteps_learned) + '_' + str(self.agent.get_wrapper_attr('state_mode')) + '_' + str(self.agent.get_wrapper_attr('observation_mode')) + '_' + str(self.agent.get_wrapper_attr('reward_mode')) + '_' + str(self.agent.get_wrapper_attr('action_mode'))
        self.model.save(folder + filename)

    def load(self, filename):
        # Load the trained agent
        if self.model_name == "DQN": self.model = DQN.load(filename, self.agent)
        elif self.model_name == "PPO": self.model = PPO.load(filename, self.agent)
        elif self.model_name == "A2C": self.model = A2C.load(filename, self.agent)
        # elif self.model_name == "BSL": self.model = BaselineModel(self.agent)
        # elif self.model_name == "TQL": self.model = QLearning.load(filename, self.agent)