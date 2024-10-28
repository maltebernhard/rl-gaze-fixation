from datetime import datetime
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm
import wandb
import yaml

from agent.agents.mixture_agent import MixtureAgent
from agent.agents.policy_agent import PolicyAgent
from agent.structure_agent import StructureAgent
from utils.callback import BaseAgentCallback
from utils.plotting import plot_actions_observations, plot_subagent_training_progress
from utils.user_interface import prompt_folder_selection
from tqdm import tqdm

# =============================================================================

class BaseAgent:
    def __init__(self, agent_config, env_config, folder=None) -> None:
        self.agent_config = agent_config
        self.env_config = env_config
        self.make_base_env(env_config)

        self.timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M')
        self.name = self.agent_config["name"].replace(" ", "-")
        self.callback = BaseAgentCallback(self.name, observation_keys=list(self.observations.keys()))
        self.parse_agents()
        self.training = False
        if folder is None:
            folder = f"./training_data/{self.timestamp.split('_')[0]}/{self.timestamp.split('_')[1]}_{self.name}/"
        self.folder = folder

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, list]:
        return self.last_agent.predict_transformed_action(observation)

    def save(self):
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
        episode_rewards_serializable = {"base": [arr.tolist() for arr in self.callback.episode_rewards]}
        for agent in self.trainable_agents:
            episode_rewards_serializable[agent.id] = [float(reward) for reward in agent.callback.episode_rewards]
        with open(folder + 'episode_rewards.yaml', 'w') as file:
            yaml.dump(episode_rewards_serializable, file, default_flow_style=False)
        self.callback.plot_training_progress(False, folder)
        plot_subagent_training_progress([agent.callback for agent in self.trainable_agents], False, folder)
        for agent in self.trainable_agents:
            plot_actions_observations(agent, num_logs=20, env_seed=5, savepath=folder)
        self.visualize_action_field()

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
                reward_dict = yaml.load(file, Loader=yaml.SafeLoader)
                base_agent.callback.episode_rewards = np.array(reward_dict["base"])
                for agent in base_agent.trainable_agents:
                    agent.callback.episode_rewards = np.array(reward_dict[agent.id])
        base_agent.folder = folder
        return base_agent

    def learn(self, total_timesteps: int, timesteps_per_run: int, save=True, plot=False) -> None:
        self.init_wandb_run(project_name="TestProject", run_name=f"{self.timestamp}_{self.name}", group=self.name, tags=["test"])
        timesteps_trained: int = 0
        
        total_runs = int(np.ceil(total_timesteps / timesteps_per_run))
        run: int = 0
        if len(self.trainable_agents) > 1:
            progress_bar = tqdm(total=total_runs+1, desc="Training Progress", position=1, leave=True, dynamic_ncols=True)
        self.training = True
        try:
            if len(self.trainable_agents) == 1:
                self.learn_agent(self.trainable_agents[0].id, total_timesteps)
            else:
                while timesteps_trained < total_timesteps:
                    for agent in self.trainable_agents:
                        run += 1
                        # test if the agent's model is a subclass of stable_baselines3.BaseAlgorithm
                        tqdm.write(f"{agent.id}: Training run {run} / {total_runs}")
                        self.callback.set_submodel_name(agent.id)
                        agent.learn(timesteps_per_run)
                        timesteps_trained += timesteps_per_run
                        progress_bar.update(1)
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            raise e
        self.training = False
        if len(self.trainable_agents) > 1:
            progress_bar.close()
        if save:
            self.save()
        if plot:
            self.callback.plot_training_progress(True)
            plot_subagent_training_progress(self.trainable_agents)
            for agent in self.trainable_agents:
                plot_actions_observations(agent, num_logs=20, env_seed=5)
        self.callback.end_training()
        self.finish_wandb_run()

    def run(self, timesteps = 0, env_seed = None, prints = False, render = True, video_path = None) -> None:
        total_reward = 0
        step = 0
        obs, info = self.reset(seed=env_seed, video_path=video_path)
        done = False
        while not done and (timesteps==0 or step < timesteps):
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
            if render:
                self.render()
        self.close()
        obs, info = self.reset()
        if prints or render:
            print(f"Episode finished with total reward {total_reward}")

    def learn_agent(self, agent: int, timesteps: int, plot=False) -> None:
        if agent in self.agents.keys():
            self.agents[agent].learn(timesteps)
            if plot:
                self.callback.plot_training_progress()
        else:
            raise ValueError(f"Invalid agent key: {agent}")
        
    def run_agent(self, agent_id: str, timesteps: int = 0, env_seed = None, render = True, prints: bool = False) -> None:
        if agent_id in self.agents.keys():
            return self.agents[agent_id].run(timesteps=timesteps, env_seed=env_seed, render=render, prints=prints)
        else:
            raise ValueError(f"Invalid agent key: {agent_id}")

    def parse_agents(self):
        agents_config: Dict[str, List[dict]] = self.agent_config["agents"]
        self.agents: dict[int, StructureAgent] = {}
        for agent_config in agents_config:
            if agent_config["type"] == "PLCY":
                self.agents[agent_config["id"]] = PolicyAgent(
                    base_agent = self,
                    agent_config = agent_config,
                )
                print(f"Policy agent {agent_config['id']} created")
            elif agent_config["type"] == "MXTR":
                self.agents[agent_config["id"]] = MixtureAgent(
                    base_agent = self,
                    agent_config = agent_config,
                    experts = [self.agents[expert] for expert in agent_config["experts"]]
                )
                print(f"New-Mixture-of-Experts agent {agent_config['id']} created")
            else:
                raise ValueError(f"Unknown model type: {agent_config['type']}")  
        self.set_last_agent()
        for agent in self.agents.values():
            agent.env.unwrapped.set_base_agent(self)
        self.trainable_agents = [agent for agent in self.agents.values() if isinstance(agent.model, BaseAlgorithm)]

    def make_base_env(self, env_config: dict):
        self.base_env = gym.make(
            id='GazeFixEnv',
            config = env_config
        )
        self.observations = self.base_env.get_wrapper_attr("observations")
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.last_observation = None

    def set_last_agent(self) -> None:
        for agent_id, agent in self.agents.items():
            is_last_agent = True
            for other_agent in self.agents.values():
                if other_agent.config["type"] == "CONT" and other_agent.config["contingent_agent"] == agent_id:
                    is_last_agent = False
                    break
                #elif (other_agent.config["type"] == "MXTR" or other_agent.config["type"] == "MX2R" or other_agent.config["type"] == "NEW_MXTR") and (agent_id in other_agent.config["experts"]):
                elif other_agent.config["type"] == "MXTR" and agent_id in other_agent.config["experts"]:
                    is_last_agent = False
                    break
            if is_last_agent:
                self.last_agent = agent
                return
            
    def init_wandb_run(self, project_name: str, run_name: str, group: str, tags: List[str]) -> None:
        self.wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config=self.agent_config,
            group=group,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=False,        # optional
            tags=tags,
        )
        self.callback.run = self.wandb_run
        for agent in self.agents.values():
            agent.callback.run = self.wandb_run

    def finish_wandb_run(self) -> None:
        # TODO: log model
        self.wandb_run.finish()
        self.callback.run = None

    # =========================== env control ===================================

    def act(self):
        action = self.predict(self.last_observation)[0]
        self.last_observation, rewards, done, truncated, info = self.step(action)
        if self.training:
            self.callback.log(action, self.last_observation, rewards, done)
        return self.last_observation, rewards, done, truncated, info

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

    def visualize_action_field(self):
        observation_field = self.base_env.unwrapped.get_observation_field()
        action_field = [[None]*len(observation_field[0]) for _ in range(len(observation_field))]
        for i in range(len(observation_field)):
            for j in range(len(observation_field[i])):
                action_field[i][j] = self.predict(observation_field[i][j])[0]
        self.base_env.unwrapped.draw_action_field(action_field, savepath=self.folder)

    def visualize_agent_tree(self, filename: str = None) -> None:
        def add_node(agent: StructureAgent, G: nx.DiGraph, depth: int = 0):
            agent_type = agent.config["type"]
            model_type = agent.config["model_type"]
            G.add_node(agent.id, agent_type=agent_type, model_type=model_type, id=agent.id)
            #if agent_type == "MXTR" or agent_type == "MX2R" or agent_type == "NEW_MXTR":
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
            #if G.nodes[node]["agent_type"] == "MXTR" or G.nodes[node]["agent_type"] == "MX2R" or G.nodes[node]["agent_type"] == "NEW_MXTR":
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
        if graph_depth > 1:
            set_parent_positions(self.last_agent.id, pos, G)

        nx.draw_networkx(G, pos, with_labels=False, node_color=[node_color_map[data.get('agent_type', '')] for _, data in G.nodes(data=True)], node_size=5000, alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight="bold")
        #nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.5, arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color="red")

        plt.axis("off")
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()