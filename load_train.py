

from agent.base_agent import BaseAgent


agent = BaseAgent.load()

#agent.learn(0, 2048, save=True, plot=True)

for i in range(10):
    agent.run(prints=True, steps=1000)