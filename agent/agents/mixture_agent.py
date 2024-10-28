import gymnasium as gym
import numpy as np
from typing import List, Tuple
from agent.structure_agent import StructureAgent

# =========================================================================================================

class MixtureAgent(StructureAgent):
    models = None

    def __init__(self, base_agent, agent_config, experts):


        self.reduce_action_space_size = True
        self.normalize = True


        super().__init__(base_agent, agent_config)
        self.experts: List[StructureAgent] = experts
        self.mixture_mode = agent_config["mixture_mode"]
        self.compute_sub_mixtures()
        self.initialize()

    # ---------------------------------- initialization -----------------------------------

    def create_mixture_matrix(self):
        mixture_matrix = -np.ones((len(self.experts), len(self.output_action_keys)), dtype=int)
        for i, expert in enumerate(self.experts):
            for j, key in enumerate(self.output_action_keys):
                if key in expert.output_action_keys:
                    mixture_matrix[i, j] = expert.output_action_keys.index(key)
        return mixture_matrix

    def compute_sub_mixtures(self):
        self.mixture_matrix = self.create_mixture_matrix()
        sub_mixtures = []
        if self.mixture_mode == 1:
            action_key_indices = list(range(len(self.output_action_keys)))
            while action_key_indices:
                action_key_index = action_key_indices.pop(0)
                identical_keys = [action_key for action_key in action_key_indices if np.array_equal(
                    np.where(self.mixture_matrix[:, action_key_index] != -1),
                    np.where(self.mixture_matrix[:, action_key] != -1)
                )]
                for action_key in identical_keys:
                    action_key_indices.remove(action_key)
                expert_indices = np.where(self.mixture_matrix[:, action_key_index] != -1)[0].tolist()
                sub_mixtures.append(([action_key_index] + identical_keys, expert_indices))
        elif self.mixture_mode == 2:
            for i in range(self.mixture_matrix.shape[1]):
                expert_indices = np.where(self.mixture_matrix[:, i] != -1)[0].tolist()
                sub_mixtures.append(([i], expert_indices))
        elif self.mixture_mode == 3:
            return
        else:
            raise ValueError("Mixture mode not supported")
        self.sub_mixtures: List[Tuple[List[int], List[int]]] = sub_mixtures

    def create_action_space(self):
        if self.mixture_mode == 3:
            return gym.spaces.MultiDiscrete([len(var[1]) for var in self.sub_mixtures])
        else:
            action_space_size = 0
            self.weight_indices = []
            self.model_action_keys = []
            for sub_mixture in self.sub_mixtures:
                if self.reduce_action_space_size and len(sub_mixture[1]) == 1:
                    self.weight_indices.append(np.array([], dtype=int))
                else:
                    for expert in sub_mixture[1]:
                        self.model_action_keys.append(f"{self.experts[expert].id}_relevance")
                    self.weight_indices.append(np.array([action_space_size + i for i in range(len(sub_mixture[1]))], dtype=int))
                    action_space_size += len(sub_mixture[1])
            return gym.spaces.Box(
                low=np.array([0.0 for _ in range(action_space_size)]).flatten(),
                high=np.array([1.0 for _ in range(action_space_size)]).flatten(),
                dtype=np.float64
            )
        
    # ---------------------------------- prediction -----------------------------------

    def transform_action(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        # get mixture experts' actions
        expert_actions = np.zeros((len(self.experts), len(self.output_action_keys)))
        for i, expert in enumerate(self.experts):
            expert_action = expert.predict_transformed_action(observation)[0]
            for j, key in enumerate(expert.output_action_keys):
                action_index = self.output_action_keys.index(key)
                expert_actions[i, action_index] = expert_action[j]

        mixture_weights = self.compute_mixture_weights(action)
    
        # print("Action:\n", action)
        # print("Mixture weights:\n", mixture_weights)
        # print("Expert actions:\n", expert_actions)
        # print("Mixture:\n", np.sum(expert_actions * mixture_weights, axis = 0))

        return np.sum(expert_actions * mixture_weights, axis = 0)
    
    def compute_mixture_weights(self, weights):
        weight_matrix = np.zeros((len(self.experts), len(self.output_action_keys)))
        for i, sub_mixture in enumerate(self.sub_mixtures):
            output_action_dimensions, experts = sub_mixture
            if len(experts) == 0:
                raise Exception("Something's wrong")
            elif self.reduce_action_space_size and len(experts) == 1:
                weight_matrix[experts[0]][output_action_dimensions] = 1.0
            else:
                normalized_weights = self.normalize_weights(weights[self.weight_indices[i]], len(output_action_dimensions), experts)
                for j, expert in enumerate(experts):
                    for k, dim in enumerate(output_action_dimensions):
                        weight_matrix[expert][dim] = normalized_weights[j][k]
        return weight_matrix

    def new_normalize_weights(self, weights: np.ndarray, subaction_dimension, experts) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = np.concatenate([weights, len(weights)-np.sum(weights)]) / len(weights)
            weights = weights.reshape((len(experts),1)).repeat(subaction_dimension, axis=1)
        else:
            weights = weights.reshape((len(experts),subaction_dimension))
            weights = np.concatenate([weights, (len(experts)-1)*np.ones(subaction_dimension)-np.sum(weights, axis=0)]) / len(experts)-1
        return weights

    def normalize_weights(self, weights: np.ndarray, subaction_dimensionality, experts) -> np.ndarray:
        if self.mixture_mode == 1:
            weights = weights.reshape((len(experts),1)).repeat(subaction_dimensionality, axis=1)
        else:
            weights = weights.reshape((len(experts),subaction_dimensionality))
        
        # normalize weights only where the sum is larger than 1.0
        if self.normalize:
            sum = np.sum(weights, axis=0)
            weights[:, sum == 0] = 1 / weights.shape[0]
            sum = np.sum(weights, axis=0)
            # normalizfind out why we need to normalize all weights
            weights = weights / sum

        return weights