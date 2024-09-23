from datetime import datetime
import numpy as np
import gymnasium as gym
from typing import Dict, List
from stable_baselines3.common.utils import set_random_seed
import yaml

from agent.agent import Contingency, MixtureOfExperts, Policy, StructureAgent
from utils.plotting import plot_training_progress
import matplotlib.pyplot as plt
import networkx as nx


class BaseAgent:
    def __init__(self, agent_config, env_config) -> None:
        self.agent_config = agent_config
        self.env_config = env_config
        self.base_env = self.make_base_env(env_config)
        self.parse_agents(self.base_env)
        self.reset()

    def make_base_env(self, env_config: dict):
        env = gym.make(
            id='GazeFixEnv',
            config = env_config
        )
        base_env = gym.make(
            id='BaseEnv',
            env = env,
        )
        base_env.set_base_agent(self)
        return base_env

    def set_last_agent(self, last_agent) -> None:
        self.last_agent: StructureAgent = last_agent

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self.last_agent.transform_action(self.last_agent.predict_full_observation(observation)[0], observation)
    
    def reset(self) -> None:
        set_random_seed(self.agent_config["random_seed"])

    def save(self, folder = None):
        if folder is None:
            folder = "./training_data/" + datetime.today().strftime('%Y-%m-%d_%H-%M') + "/"
        for id, agent in self.agents.items():
            filename = f"{id}_model"
            agent.model.save(folder + filename)
        self.visualize_agent_tree(folder + "agent_tree.png")
        with open(folder + 'env_config.yaml', 'w') as file:
            yaml.dump(self.env_config, file, default_flow_style=False)
        with open(folder + 'agent_config.yaml', 'w') as file:
            yaml.dump(self.agent_config, file, default_flow_style=False)

    def load(self):
        pass

    def parse_agents(self, base_env):
        agents_config: Dict[int, dict] = self.agent_config["agents"]
        self.agents: dict[int, StructureAgent] = {}
        for agent_config in agents_config:
            if agent_config["type"] == "PLCY":
                self.agents[agent_config["id"]] = Policy(
                    base_env = base_env,
                    agent_config = agent_config
                )
            elif agent_config["type"] == "CONT":
                self.agents[agent_config["id"]] = Contingency(
                    base_env = base_env,
                    agent_config = agent_config,
                    contingent_agent = self.agents[agent_config["contingent_agent"]]
                )
            elif agent_config["type"] == "MXTR":
                self.agents[agent_config["id"]] = MixtureOfExperts(
                    base_env = base_env,
                    agent_config = agent_config,
                    experts = [self.agents[expert] for expert in agent_config["experts"]]
                )
            else:
                raise ValueError(f"Unknown model type: {agent_config['type']}")
        key = max(self.agents.keys())
        self.set_last_agent(self.agents[key])
        for agent in self.agents.values():
            agent.env.reset()

    def learn_agent(self, agent: int, timesteps: int, plot=False) -> None:
        if agent in self.agents.keys():
            self.agents[agent].learn(timesteps)
            if plot:
                plot_training_progress(self.agents[agent].callback)
        else:
            raise ValueError(f"Invalid agent key: {agent}")
        
    def run_agent(self, agent: int, timesteps: int, prints: bool = False) -> None:
        if agent in self.agents.keys():
            self.agents[agent].run(prints=prints, steps=timesteps)
        else:
            raise ValueError(f"Invalid agent key: {agent}")

    # ========================== visualization ==================================

    def visualize_agent_tree(self, filename: str) -> None:
        def add_node(agent: StructureAgent, G: nx.DiGraph, depth: int = 0):
            agent_type = agent.config["type"]
            model_type = agent.config["model_type"]
            G.add_node(agent.id, agent_type=agent_type, model_type=model_type, id=agent.id)
            if agent_type == "MXTR":
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
            if G.nodes[node]["agent_type"] == "MXTR":
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
        set_parent_positions(self.last_agent.id, pos, G)

        nx.draw_networkx(G, pos, with_labels=False, node_color=[node_color_map[data.get('agent_type', '')] for _, data in G.nodes(data=True)], node_size=5000, alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight="bold")
        #nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.5, arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

        plt.axis("off")
        plt.savefig(filename)
        plt.close()