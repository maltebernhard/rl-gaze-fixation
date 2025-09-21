from agent.base_agent import BaseAgent

if __name__ == "__main__":
    agent = BaseAgent.load(new_learning_rate=0.0001)
    agent.learn(100000, 2048, save=True, plot=True)