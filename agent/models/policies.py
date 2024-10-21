import numpy as np
from agent.base_model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class AvoidNearestObstacleModel(Model):
    id = "ANO"
    observation_keys = [item for sublist in [[f"obstacle{i+1}_offset_angle", f"obstacle{i+1}_coverage"] for i in range(100)] for item in sublist]
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.base_env_config["num_obstacles"]
        self.action_space_dimensionality = self.env.action_space.shape[0]

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
    id = "A1O"
    observation_keys = [f"obstacle1_offset_angle"]
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

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
    
# =======================================================================================================
    
class AvoidObstacle2Model(Model):
    id = "A2O"
    observation_keys = [f"obstacle2_offset_angle"]
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

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
    
# =======================================================================================================

class AvoidObstacle3Model(Model):
    id = "A3O"
    observation_keys = [f"obstacle3_offset_angle"]
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]
    
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
    
# =======================================================================================================

class KeepTargetDistanceModel(Model):
    id = "KTD"
    observation_keys = ["target_offset_angle", "del_target_offset_angle", "vel_frontal", "vel_lateral", "vel_rot"]
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.action_mode = self.env.unwrapped.action_mode
        self.timestep = self.env.unwrapped.timestep
        self.state = None
        self.action = None

        self.target_distance = self.env.unwrapped.config["target_distance"]

        self.max_acc = self.env.unwrapped.robot.max_acc

    def estimate_distance(self, state):
        beta = abs(state[1]*self.timestep + state[0]-self.state[0])
        b = np.sqrt((state[2]*self.timestep)**2 + (state[3]*self.timestep)**2)

        vel = np.array([state[2], state[3]])
        del_orientation = self.state[1]*self.timestep
        vel = np.array([[np.cos(-del_orientation), -np.sin(-del_orientation)], [np.sin(-del_orientation), np.cos(-del_orientation)]]) @ vel
        alpha =  abs(np.arctan2(vel[1]*self.timestep, vel[0]*self.timestep))

        gamma = np.pi - alpha - beta
        estimated_distance = b * np.sin(gamma) / np.sin(beta) - self.target_distance
        #print(f"alpha: {alpha}\nb: {b}\nbeta: {beta}\nEstimate: {estimated_distance}")
        return np.concatenate([state,np.array([estimated_distance])])

    # TODO: reprogram to fix
    def predict(self, state, eps = 0.01, deterministic: bool = True):
        if self.state is None:
            self.state = state
            
        #state = self.estimate_distance(state)

        self.state = state

        acc_lateral = 1

        if self.state[5] > eps: acc_frontal = 2
        elif self.state[5] < 0:
            acc_frontal = 1
            if self.state[3] > 0:
                acc_lateral = 0
        else:
            acc_frontal = 1
            if self.state[3] > 0:
                acc_lateral = 0
            elif self.state[3] < 0:
                acc_frontal = 2

        self.action = np.array([2, acc_lateral, 1])
        if self.action_mode == 1:
            self.action = (self.action - np.ones(self.action.shape)) * self.max_acc
        return self.action, []
    
# =======================================================================================================

class StandStillModel(Model):
    id = "SST"
    observation_keys = []
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.action_space_dimensionality = self.env.action_space.shape[0]

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        action = np.array([0.0]*self.action_space_dimensionality)
        return action, None
    
# =======================================================================================================

class GoLeftModel(Model):
    id = "GLM"
    observation_keys = []
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.action_space_dimensionality = self.env.action_space.shape[0]

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        if self.action_space_dimensionality == 2:
            action = np.array([0.0, 1.0])
        elif self.action_space_dimensionality == 3:
            action = np.array([0.0, 1.0, 0.0])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None
    
# =======================================================================================================

class GoRightModel(Model):
    id = "GRM"
    observation_keys = []
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.action_space_dimensionality = self.env.action_space.shape[0]

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        if self.action_space_dimensionality == 2:
            action = np.array([0.0, -1.0])
        elif self.action_space_dimensionality == 3:
            action = np.array([0.0, -1.0, 0.0])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None
    
# =======================================================================================================

class TowardsTargetModel(Model):
    id = "GTT"
    observation_keys = ["target_offset_angle"]
    action_keys = ["frontal movement", "lateral movement", "rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.action_space_dimensionality = self.env.action_space.shape[0]

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        offset_angle = state[0]
        # generate unit length vector in the direction of the target
        if self.action_space_dimensionality == 2:
            action = np.array([np.cos(offset_angle), np.sin(offset_angle)])
        elif self.action_space_dimensionality == 3:
            action = np.array([np.cos(offset_angle), np.sin(offset_angle)] + [np.random.uniform(-1,1)])
        else:
            raise ValueError("Invalid action space dimensionality")
        return action, None
    
# =======================================================================================================

