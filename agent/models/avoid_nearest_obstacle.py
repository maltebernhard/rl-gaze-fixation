import numpy as np
from agent.models.model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class AvoidNearestObstacleModel(Model):
    observation_keys = [item for sublist in [[f"obstacle{i+1}_offset_angle", f"obstacle{i+1}_coverage"] for i in range(100)] for item in sublist]

    def __init__(self, env: StructureEnv, action_space_dimensionality: int):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.base_agent.base_env.unwrapped.config["num_obstacles"]
        self.action_space_dimensionality = action_space_dimensionality

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        nearest_obstacle = 0
        for i in range(1,self.num_obstacles):
            if state[2*i+1] > state[2*nearest_obstacle+1]:
                nearest_obstacle = i
        offset_angle = state[2*nearest_obstacle]
        # create a vector of length one in opposite direction of the obstacle
        if self.action_space_dimensionality == 3:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        elif self.action_space_dimensionality == 2:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None
    
# =======================================================================================================
    
class AvoidObstacle1Model(Model):
    observation_keys = [f"obstacle1_offset_angle"]
    def __init__(self, env: StructureEnv, action_space_dimensionality: int):
        super().__init__(env)
        self.action_space_dimensionality = action_space_dimensionality
    def predict(self, state, eps = 0.01, deterministic: bool = True):                
        offset_angle = state[0]
        # create a vector of length one in opposite direction of the obstacle
        if self.action_space_dimensionality == 3:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        elif self.action_space_dimensionality == 2:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None
    
class AvoidObstacle2Model(Model):
    observation_keys = [f"obstacle2_offset_angle"]
    def __init__(self, env: StructureEnv, action_space_dimensionality: int):
        super().__init__(env)
        self.action_space_dimensionality = action_space_dimensionality
    def predict(self, state, eps = 0.01, deterministic: bool = True):                
        offset_angle = state[0]
        # create a vector of length one in opposite direction of the obstacle
        if self.action_space_dimensionality == 3:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        elif self.action_space_dimensionality == 2:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None
    
class AvoidObstacle3Model(Model):
    observation_keys = [f"obstacle3_offset_angle"]
    def __init__(self, env: StructureEnv, action_space_dimensionality: int):
        super().__init__(env)
        self.action_space_dimensionality = action_space_dimensionality
    def predict(self, state, eps = 0.01, deterministic: bool = True):                
        offset_angle = state[0]
        # create a vector of length one in opposite direction of the obstacle
        if self.action_space_dimensionality == 3:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        elif self.action_space_dimensionality == 2:
            action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None