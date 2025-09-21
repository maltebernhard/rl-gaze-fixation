from agent.base_agent import BaseAgent

if __name__ == "__main__":
    agent = BaseAgent.load()
    agent.run_agent("Mixture", timesteps=1000, env_seed=12, prints=True)