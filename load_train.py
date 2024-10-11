

from agent.base_agent import BaseAgent


agent = BaseAgent.load(new_learning_rate=0.0001)

agent.learn(100000, 2048, save=True, plot=True)