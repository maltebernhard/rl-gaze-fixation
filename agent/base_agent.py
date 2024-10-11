from datetime import datetime
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List
from stable_baselines3.common.base_class import BaseAlgorithm
from wandb.integration.sb3 import WandbCallback
import yaml

from agent.structure_agent import StructureAgent
from agent.agents.contingency import Contingency
from agent.agents.mixture import MixtureOfExperts
from agent.agents.mix2re import MixtureOfTwoExperts
from agent.agents.policy import Policy
from utils.callback import ModularAgentCallback
from utils.user_interface import prompt_folder_selection
from tqdm import tqdm

# =============================================================================

class BaseAgent:
    def __init__(self, agent_config, env_config) -> None:
        self.agent_config = agent_config
        self.env_config = env_config
        self.make_base_env(env_config)

        self.callback = ModularAgentCallback(model_name=self.agent_config["name"])
        self.name = self.agent_config["name"].replace(" ", "-")
        self.reward_indices = self.agent_config["reward_indices"] if "reward_indices" in self.agent_config.keys() else [0]
        self.parse_agents()
        self.training = False
        self.folder = None

    def predict(self, observation: np.ndarray, rewards=None) -> np.ndarray:
        if rewards is not None:
            self.callback.current_base_episode_reward += np.sum(rewards[self.reward_indices])
        return self.last_agent.transform_action(self.last_agent.predict_full_observation(observation)[0], observation), []

    def save(self):
        if self.folder is None:
            self.folder = "./training_data/" + datetime.today().strftime('%Y-%m-%d/%H-%M') + f"_{self.name}/"
        folder = self.folder
        for id, agent in self.agents.items():
            filename = f"{id}_model"
            agent.model.save(folder + filename)
        self.visualize_agent_tree(folder + "agent_tree")
        with open(folder + 'env_config.yaml', 'w') as file:
            yaml.dump(self.env_config, file, default_flow_style=False)
        with open(folder + 'agent_config.yaml', 'w') as file:
            yaml.dump(self.agent_config, file, default_flow_style=False)
        # Convert numpy arrays to lists before saving
        episode_rewards_serializable = {k: [arr.tolist() for arr in v] for k, v in self.callback.episode_rewards.items()}
        with open(folder + 'episode_rewards.yaml', 'w') as file:
            yaml.dump(episode_rewards_serializable, file, default_flow_style=False)
        self.callback.plot_training_progress(False, folder)
        self.callback.plot_subagent_training_progress(False, folder)

    @classmethod
    def load(self, folder = None, new_learning_rate = None):
        if folder is None:
            folder = prompt_folder_selection()
        with open(folder + 'env_config.yaml', 'r') as file:
            env_config = yaml.load(file, Loader=yaml.SafeLoader)
        with open(folder + 'agent_config.yaml', 'r') as file:
            agent_config = yaml.load(file, Loader=yaml.SafeLoader)
        if new_learning_rate is not None:
            for i, agent in enumerate(agent_config["agents"]):
                if "learning_rate" in agent.keys():
                    agent_config["agents"][i]["learning_rate"] = new_learning_rate
        base_agent = BaseAgent(agent_config, env_config)
        for id, agent in base_agent.agents.items():
            filename = f"{id}_model"
            agent.load(folder + filename)
        if 'episode_rewards.yaml' in os.listdir(folder):
            with open(folder + 'episode_rewards.yaml', 'r') as file:
                base_agent.callback.episode_rewards = yaml.load(file, Loader=yaml.SafeLoader)
            # Convert lists back to numpy arrays
            base_agent.callback.episode_rewards = {k: [np.array(arr) for arr in v] for k, v in base_agent.callback.episode_rewards.items()}
        base_agent.folder = folder
        return base_agent

    def learn(self, total_timesteps: int, timesteps_per_run: int, save=True, plot=False) -> None:
        timesteps: int = 0
        trainable_agents = [agent for agent in self.agents.values() if isinstance(agent.model, BaseAlgorithm)]
        total_runs = int(np.ceil(total_timesteps / timesteps_per_run / len(trainable_agents)))
        run: int = 0
        if len(trainable_agents) > 1:
            progress_bar = tqdm(total=total_runs, desc="Training Progress", position=0, leave=True, dynamic_ncols=True)
        self.training = True
        try:
            if len(trainable_agents) == 1:
                self.learn_agent(trainable_agents[0].id, total_timesteps)
            else:
                while timesteps < total_timesteps:
                    for agent in trainable_agents:
                        run += 1
                        # test if the agent's model is a subclass of stable_baselines3.BaseAlgorithm
                        tqdm.write(f"{agent.id}: Training run {run} / {total_runs}")
                        self.callback.set_submodel_name(agent.id)
                        agent.learn(timesteps_per_run)
                        timesteps += timesteps_per_run
                        progress_bar.update(1)
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            raise e
        self.training = False
        if len(trainable_agents) > 1:
            progress_bar.close()
        if save:
            self.save()
        if plot:
            self.callback.plot_training_progress(True)
            self.callback.plot_subagent_training_progress(True)

    def run(self, prints = False, steps = 0, env_seed = None) -> None:
        total_reward = 0
        step = 0
        obs, info = self.reset(seed=env_seed)
        done = False
        while not done and (steps==0 or step < steps):
            action, _states = self.predict(obs)
            if prints:
                print(f'-------------------- Step {step} ----------------------')
                print(f'Observation: {obs}')
                print(f'Action:      {action}')
            obs, reward, done, truncated, info = self.step(action)
            if prints:
                print(f'Reward:      {reward}')
            total_reward += reward
            step += 1
            self.render()
        self.close()
        obs, info = self.reset()
        print(f"Episode finished with total reward {total_reward}")

    def learn_agent(self, agent: int, timesteps: int, plot=False) -> None:
        if agent in self.agents.keys():
            self.callback.set_submodel_name(agent)
            self.agents[agent].learn(timesteps)
            if plot:
                self.callback.plot_training_progress()
        else:
            raise ValueError(f"Invalid agent key: {agent}")
        
    def run_agent(self, agent: int, timesteps: int, prints: bool = False, env_seed = None, render=True) -> None:
        if agent in self.agents.keys():
            return self.agents[agent].run(prints=prints, steps=timesteps, env_seed=env_seed, render=render)
        else:
            raise ValueError(f"Invalid agent key: {agent}")

    def parse_agents(self):
        agents_config: Dict[int, dict] = self.agent_config["agents"]
        self.agents: dict[int, StructureAgent] = {}
        for agent_config in agents_config:
            if agent_config["type"] == "PLCY":
                self.agents[agent_config["id"]] = Policy(
                    base_agent = self,
                    agent_config = agent_config,
                    callback = self.callback
                )
                print(f"Policy agent {agent_config['id']} created")
            elif agent_config["type"] == "CONT":
                self.agents[agent_config["id"]] = Contingency(
                    base_agent = self,
                    agent_config = agent_config,
                    callback = self.callback,
                    contingent_agent = self.agents[agent_config["contingent_agent"]]
                )
                print(f"Contingency agent {agent_config['id']} created")
            elif agent_config["type"] == "MXTR":
                self.agents[agent_config["id"]] = MixtureOfExperts(
                    base_agent = self,
                    agent_config = agent_config,
                    callback = self.callback,
                    experts = [self.agents[expert] for expert in agent_config["experts"]]
                )
                print(f"Mixture-of-Experts agent {agent_config['id']} created")
            elif agent_config["type"] == "MX2R":
                self.agents[agent_config["id"]] = MixtureOfTwoExperts(
                    base_agent = self,
                    agent_config = agent_config,
                    callback = self.callback,
                    experts = [self.agents[expert] for expert in agent_config["experts"]]
                )
                print(f"Mixture-of-Two-Experts agent {agent_config['id']} created")
            else:
                raise ValueError(f"Unknown model type: {agent_config['type']}")  
        self.set_last_agent()
        for agent in self.agents.values():
            agent.env.set_base_agent(self)

    def make_base_env(self, env_config: dict):
        self.base_env = gym.make(
            id='GazeFixEnv',
            config = env_config
        )
        self.observations = self.base_env.get_wrapper_attr("observations")
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.last_observation = None
        self.last_rewards = None

    def set_last_agent(self) -> None:
        for agent_id, agent in self.agents.items():
            is_last_agent = True
            for other_agent in self.agents.values():
                if other_agent.config["type"] == "CONT" and other_agent.config["contingent_agent"] == agent_id:
                    is_last_agent = False
                    break
                elif (other_agent.config["type"] == "MXTR" or other_agent.config["type"] == "MX2R") and (agent_id in other_agent.config["experts"]):
                    is_last_agent = False
                    break
            if is_last_agent:
                self.last_agent = agent
                return
            
    def set_wandb_callback(self):
        for agent in self.agents.values():
            agent.callback = WandbCallback(
                gradient_save_freq=100,
                model_save_path=None,
                verbose=2,
            )

    # =========================== env control ===================================

    def act(self):
        action = self.predict(self.last_observation, self.last_rewards)[0]
        self.last_observation, self.last_rewards, done, truncated, info = self.step(action)
        return self.last_observation, self.last_rewards, done, truncated, info

    def step(self, action: np.ndarray):
        return self.base_env.step(action)

    def reset(self, seed=None, **kwargs):
        self.last_observation, info = self.base_env.reset(seed=seed, **kwargs)
        return self.last_observation, info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()

    # ========================== visualization ==================================

    def visualize_agent_tree(self, filename: str) -> None:
        def add_node(agent: StructureAgent, G: nx.DiGraph, depth: int = 0):
            agent_type = agent.config["type"]
            model_type = agent.config["model_type"]
            G.add_node(agent.id, agent_type=agent_type, model_type=model_type, id=agent.id)
            if agent_type == "MXTR" or agent_type == "MX2R":
                for expert in agent.experts:
                    depth = max(depth, add_node(expert, G, depth + 1))
                    G.add_edge(expert.id, agent.id)
            elif agent_type == "CONT":
                depth = max(depth, add_node(agent.contingent_agent, G, depth + 1))
                G.add_edge(agent.contingent_agent.id, agent.id)
            return depth

        G = nx.DiGraph()
        graph_depth = add_node(self.last_agent, G)

        node_labels = {node: f"{data.get('agent_type', '')}\n{data.get('model_type', '')}" for node, data in G.nodes(data=True)}
        edge_labels = {(u, v): "" for u, v in G.edges()}
        edge_colors = ["black" if G.nodes[u].get("agent_type", "") != "MXTR" else "red" for u, v in G.edges()]

        plt.figure(figsize=(12, 8))
        agent_types = set([data.get('agent_type', '') for _, data in G.nodes(data=True)])
        node_colors = [plt.cm.tab10(i) for i in range(len(agent_types))]
        node_color_map = {agent_type: color for agent_type, color in zip(agent_types, node_colors)}

        def set_parent_positions(node, pos, G: nx.DiGraph):
            if G.nodes[node]["agent_type"] == "MXTR" or G.nodes[node]["agent_type"] == "MX2R":
                # Get the parents of the node
                parents = list(G.predecessors(node))
                x, y = pos[node]
                for i,p in enumerate(parents):
                    pos[p] = np.array([x-2/(graph_depth-1), y - 1/2 + i*1/(len(parents)-1)])
                    set_parent_positions(p, pos, G)
            elif G.nodes[node]["agent_type"] == "CONT":
                parent = list(G.predecessors(node))[0]
                x, y = pos[node]
                pos[parent] = np.array([x-2/(graph_depth-1), y])
                set_parent_positions(parent, pos, G)
            
        pos = nx.spring_layout(G)

        # Adjust the position of the nodes to have children at the same height
        root = G.nodes[self.last_agent.id]
        pos[root["id"]] = np.array([1.0, 0])
        if graph_depth > 1:
            set_parent_positions(self.last_agent.id, pos, G)

        nx.draw_networkx(G, pos, with_labels=False, node_color=[node_color_map[data.get('agent_type', '')] for _, data in G.nodes(data=True)], node_size=5000, alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight="bold")
        #nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.5, arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

        plt.axis("off")
        plt.savefig(filename)
        plt.close()