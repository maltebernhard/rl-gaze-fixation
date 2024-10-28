import numpy as np
from agent.base_model import Model
from environment.structure_env import StructureEnv

# =============================================================================

class AvoidNearestObstacleModel(Model):
    id = "ANO"
    # TODO: change coverage to distance
    observation_keys = [item for sublist in [[f"obstacle{i+1}_offset_angle", f"obstacle{i+1}_coverage"] for i in range(100)] for item in sublist]
    action_keys = ["frontal_movement", "lateral_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.num_obstacles = self.env.unwrapped.base_env_config["num_obstacles"]
        if not self.env.action_space.shape[0] == 2:
            raise ValueError("Invalid action space dimensionality")
        

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        nearest_obstacle = 0
        for i in range(1,self.num_obstacles):
            if state[2*i+1] > state[2*nearest_obstacle+1]:
                nearest_obstacle = i
        offset_angle = state[2*nearest_obstacle]
        # create a vector of length one in opposite direction of the obstacle
        action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)])
        return action, None
    
# =======================================================================================================

class AvoidObstacleModel(Model):
    id = "AOM"
    observation_keys = [f"obstacle1_offset_angle", f"obstacle1_distance"]
    action_keys = ["frontal_movement", "lateral_movement", "rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        if not self.env.action_space.shape[0] == 2:
            raise ValueError("Invalid action space dimensionality")
        self.max_vel = env.unwrapped.base_env_config["robot_max_vel"]
        self.timestep = env.unwrapped.base_env_config["timestep"]
        self.evasion_distance_margin = 5.0

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        offset_angle = state[0]
        vel = max(self.evasion_distance_margin - state[1], 0.0) / self.evasion_distance_margin
        # create a vector of length one in opposite direction of the obstacle
        action = np.array([-np.cos(offset_angle), -np.sin(offset_angle)]) * vel
        return action, None
    
# =======================================================================================================

class TODOKeepTargetDistanceModel(Model):
    id = "KTD"
    observation_keys = ["target_offset_angle", "del_target_offset_angle", "vel_frontal", "vel_lateral", "vel_rot"]
    action_keys = ["frontal_movement", "lateral_movement", "rotational_movement"]

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
    action_keys = ["frontal_movement", "lateral_movement"]

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
    action_keys = ["frontal_movement", "lateral_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        if not self.env.action_space.shape[0] == 2:
            raise ValueError("Invalid action space dimensionality")

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        action = np.array([0.0, 1.0])
        return action, None
    
# =======================================================================================================

class GoRightModel(Model):
    id = "GRM"
    observation_keys = []
    action_keys = ["frontal_movement", "lateral_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        if not self.env.action_space.shape[0] == 2:
            raise ValueError("Invalid action space dimensionality")

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        action = np.array([0.0, -1.0])
        return action, None
    
# =======================================================================================================

class TowardsTargetModel(Model):
    id = "GTT"
    observation_keys = ["target_offset_angle"]
    action_keys = ["frontal_movement", "lateral_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        if not self.env.action_space.shape[0] == 2:
            raise ValueError("Invalid action space dimensionality")

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        offset_angle = state[0]
        # generate unit length vector in the direction of the target
        action = np.array([np.cos(offset_angle), np.sin(offset_angle)])
        return action, None
    
# =======================================================================================================

class ToTargetModel(Model):
    id = "TTM"
    observation_keys = ["target_offset_angle", "robot_target_distance"]
    action_keys = ["frontal_movement", "lateral_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        if not self.env.action_space.shape[0] == 2:
            raise ValueError("Invalid action space dimensionality")
        self.max_vel = env.unwrapped.base_env_config["robot_max_vel"]
        self.timestep = env.unwrapped.base_env_config["timestep"]

    def predict(self, state, eps = 0.01, deterministic: bool = True):
        offset_angle = state[0]
        vel = min(state[1] * self.max_vel * self.timestep, 1.0)
        # generate unit length vector in the direction of the target
        action = np.array([np.cos(offset_angle), np.sin(offset_angle)]) * vel
        return action, None
    
# =======================================================================================================

class NewGazeFixationModel(Model):
    id = "NGFM"
    observation_keys = ["target_offset_angle", "del_target_offset_angle"]
    action_keys = ["rotational_movement"]

    def __init__(self, env: StructureEnv):
        super().__init__(env)
        self.max_vel = env.unwrapped.base_env_config["robot_max_vel_rot"]
        self.max_acc = env.unwrapped.base_env_config["robot_max_acc_rot"]
        self.breaking_angle = (self.max_vel ** 2) / (2 * self.max_acc)
        self.action_mode = env.unwrapped.base_env_config["action_mode"]
        self.timestep = env.unwrapped.base_env_config["timestep"]
        self.epsilon = self.timestep

    def predict(self, obs, deterministic = True):
        vel_rot_desired = self.compute_target_vel(obs[0])
        if self.action_mode == 1:
            return np.array([self.pd_control(vel_rot_desired-obs[1], obs[1])]), None
        elif self.action_mode == 2:
            return np.array([self.flip_control(vel_rot_desired, obs[1])]), None
        elif self.action_mode == 3:
            return np.array([self.vel_control(obs[0])]), None
        
    def vel_control(self, target_offset_angle):
        return max(min(target_offset_angle/(self.max_vel*self.timestep), 1.0), -1.0)

    def flip_control(self, target, current):
        action = 1
        if target-current > self.epsilon:
            action = 2
        elif target-current < -self.epsilon:
            action = 0
        if self.action_mode == 1:
            return (action-1)
        return action

    # TODO: only works for continuous action
    def pd_control(self, x, del_x):
        K_p = 1.0
        K_d = 0.1
        return K_p * x - K_d * del_x
    
    def p_control(self, x):
        K_p = 10.0
        return K_p * x
    
    def compute_target_vel(self, target_angle_offset):
        if abs(target_angle_offset) > self.breaking_angle:
            vel_rot_desired = target_angle_offset/abs(target_angle_offset)
        else:
            vel_rot_desired = target_angle_offset/self.breaking_angle
        return vel_rot_desired