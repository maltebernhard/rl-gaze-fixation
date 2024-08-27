
import random
from typing import List
import gymnasium as gym
import numpy as np
import pygame
import math

# =====================================================================================================

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)

SCREEN_SIZE = 900

# =====================================================================================================

class Pose:
    def __init__(self, x = 0.0, y = 0.0, phi = 0.0):
        self.x = x
        self.y = y
        self.phi = phi

    def __add__(self, other):
        if isinstance(other, np.ndarray) and other.shape == (3,):
            return Pose(
                self.x + other[0],
                self.y + other[1],
                self.phi + other[2]
            )
        if isinstance(other, Pose):
            return Pose(
                self.x + other.x,
                self.y + other.y,
                self.phi + other.phi
            )
        else:
            return NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Pose(
                self.x * other,
                self.y * other,
                self.phi * other
            )
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __repr__(self):
        return f"Pose: x={self.x}, y={self.y}, phi={self.phi}"

class Robot:
    def __init__(self, max_vel, max_vel_rot, max_acc, max_acc_rot):
        self.pose: Pose = Pose()
        self.del_pose: Pose = Pose()
        self.sensor_angle = np.pi / 2
        self.size = 0.5
        self.max_vel = max_vel
        self.max_vel_rot = max_vel_rot
        self.max_acc = max_acc
        self.max_acc_rot = max_acc_rot

class Obstacle:
    def __init__(self, radius, pos):
        self.radius = radius
        self.pos: Pose = pos

class Target:
    def __init__(self, x=0, y=0):
        self.pose = Pose(x,y)
        self.del_pose = Pose(0,0)

# =====================================================================================================

class Environment(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.episode_length = config["episode_length"]
        self.timestep: float = config["timestep"]
        self.time = 0.0

        self.action_mode: int = config["action_mode"]

        self.distance: float = config["target_distance"]
        self.wall_collision: bool = config["wall_collision"]
        self.num_obstacles: bool = config["num_obstacles"]

        # env dimensions
        self.world_size = config["world_size"]
        self.screen_size = SCREEN_SIZE
        self.scale = SCREEN_SIZE / self.world_size

        self.robot = Robot(config["robot_max_vel"], config["robot_max_vel_rot"], config["robot_max_acc"], config["robot_max_acc_rot"])
        self.target = Target((np.random.random()-0.5)*self.world_size, (np.random.random()-0.5)*self.world_size)
        self.obstacles: List[Obstacle] = []
        for _ in range(self.num_obstacles):
            radius = np.random.random()*self.world_size/6
            self.obstacles.append(Obstacle(radius,Pose(random.choice([1, -1])*max((np.random.random())*self.world_size/2, radius), random.choice([1, -1])*max((np.random.random())*self.world_size/2, radius))))

        self.observation_space = gym.spaces.Box(
            low=np.array([-self.robot.sensor_angle/2, self.distance, 0.0]),
            high=np.array([self.robot.sensor_angle/2, self.distance, np.inf]),
            shape=(3,),
            dtype=np.float64
        )
        if self.action_mode == 1:
            self.action_space = gym.spaces.Box(
                low=np.array([-self.robot.max_acc, -self.robot.max_acc, -self.robot.max_acc_rot]),
                high=np.array([self.robot.max_acc, self.robot.max_acc, self.robot.max_acc_rot]),
                shape=(3,),
                dtype=np.float64
            )
        elif self.action_mode == 2:
            self.action_space = gym.spaces.MultiDiscrete(
                np.array([3, 3, 3])
            )
        else: raise NotImplementedError

        self.collision: bool = False

        # rendering window
        self.viewer = None
        metadata = {'render_modes': ['human'], 'render_fps': 1/self.timestep}

        self.reset(self.config["seed"])
    
    def step(self, action):
        self.time += self.timestep
        if self.action_mode == 1:
            action = self.validate_action(action) # make sure acceleration vector is within bounds
        self.move_robot(action)
        self.move_target()
        return self.get_observation(), self.get_reward(), self.get_terminated(), False, self.get_info()
    
    def reset(self, seed=None, **kwargs):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        self.time = 0.0
        self.robot = Robot(self.config["robot_max_vel"], self.config["robot_max_vel_rot"], self.config["robot_max_acc"], self.config["robot_max_acc_rot"])
        self.target = Target((np.random.random()-0.5)*self.world_size, (np.random.random()-0.5)*self.world_size)
        self.collision = False
        return self.get_observation(), self.get_info()
    
    def close(self):
        pygame.quit()
        self.screen = None
        
    def get_observation(self):
        angle = self.normalize_angle(np.arctan2(self.target.pose.y-self.robot.pose.y, self.target.pose.x-self.robot.pose.x) - self.robot.pose.phi)
        if angle>-self.robot.sensor_angle/2 and angle<self.robot.sensor_angle/2:
            return np.array([angle, self.distance, self.target_distance()])
        else:
            return np.array([np.pi, self.distance, self.target_distance()])
    
    def get_reward(self):
        if self.collision:
            return -1000000
        dist = self.target_distance()
        if abs(dist-self.distance) < 1.0:
            return 1.0 / (abs(dist-self.distance) + 1.0) * self.timestep
        return 0.0
    
    def get_terminated(self):
        return self.time > self.episode_length or self.collision
    
    def get_info(self):
        return {}
    
    # -------------------------------------- helpers -------------------------------------------

    def validate_action(self, action):
        translative_acc = action[:2]
        translative_length = np.linalg.norm(translative_acc)
        if translative_length > self.robot.max_acc:
            translative_acc = (translative_acc / translative_length) * self.robot.max_acc
        return np.array([translative_acc[0], translative_acc[1], action[2]])
    
    def move_robot(self, action):
        # set xy and angular accelerations
        if self.action_mode == 1:
            xy_acc = np.array([[np.cos(self.robot.pose.phi), -np.sin(self.robot.pose.phi)], [np.sin(self.robot.pose.phi), np.cos(self.robot.pose.phi)]]) @ np.array([action[0], action[1]])
            phi_acc = action[2]
        elif self.action_mode == 2:
            xy_acc = np.array([[np.cos(self.robot.pose.phi), -np.sin(self.robot.pose.phi)], [np.sin(self.robot.pose.phi), np.cos(self.robot.pose.phi)]]) @ np.array([(action[0]-1)*self.robot.max_acc, (action[1]-1)*self.robot.max_acc])
            phi_acc = (action[2] - 1) * self.robot.max_acc_rot
        self.robot.del_pose += np.concatenate([xy_acc, np.array([phi_acc])]) * self.timestep    # update robot velocity vector
        self.limit_robot_velocity()
        # move robot
        self.robot.pose += self.robot.del_pose * self.timestep
        # constrain phi to range [-pi, pi]
        if self.robot.pose.phi < -np.pi: self.robot.pose.phi = self.robot.pose.phi + 2*np.pi
        elif self.robot.pose.phi > np.pi: self.robot.pose.phi = self.robot.pose.phi - 2*np.pi
        self.check_collision()

    def limit_robot_velocity(self):  
        vel_vec = np.array([self.robot.del_pose.x, self.robot.del_pose.y])
        vel = np.linalg.norm(vel_vec)                                                           # compute absolute translational velocity
        if vel > self.robot.max_vel:                                                                 # make sure translational velocity is within bounds
            vel_vec = vel_vec / vel * self.robot.max_vel
        del_phi = self.robot.del_pose.phi                                                       # make sure rotational velocity is within bounds
        if abs(del_phi) > self.robot.max_vel_rot:
            del_phi = np.sign(del_phi) * self.robot.max_vel_rot
        self.robot.del_pose = Pose(vel_vec[0], vel_vec[1], del_phi)                             # update robot velocity vector

    def move_target(self):
        # TODO: implement
        pass
    
    def check_collision(self):
        for o in self.obstacles:
            if np.linalg.norm(np.array([o.pos.x-self.robot.pose.x,o.pos.y-self.robot.pose.y])) < o.radius + self.robot.size / 2:
                self.collision = True
                return
        if self.wall_collision:
            if self.robot.pose.x < -self.world_size / 2 + self.robot.size/2:
                self.robot.pose.x = -self.world_size / 2 + self.robot.size/2
                self.collision = True
            elif self.robot.pose.x > self.world_size / 2 - self.robot.size/2:
                self.robot.pose.x = self.world_size / 2 - self.robot.size/2
                self.collision = True
            if self.robot.pose.y < -self.world_size / 2 + self.robot.size/2:
                self.robot.pose.y = -self.world_size / 2 + self.robot.size/2
                self.collision = True
            elif self.robot.pose.y > self.world_size / 2 - self.robot.size/2:
                self.robot.pose.y = self.world_size / 2 - self.robot.size/2
                self.collision = True

    def target_distance(self):
        return np.linalg.norm(np.array([self.target.pose.x-self.robot.pose.x,self.target.pose.y-self.robot.pose.y]))
    
    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def calculate_circle_coverage(self, radius, distance, fov_angle_degrees):
        # Convert FOV angle from degrees to radians
        fov_angle_radians = math.radians(fov_angle_degrees)
        # Calculate the angular size of the circle
        angular_size = 2 * math.atan(radius / distance)
        # Calculate the proportion of the FOV occupied by the circle
        coverage = angular_size / fov_angle_radians
        # Ensure the coverage value is between 0 and 1
        return min(coverage, 1.0)

    # ----------------------------------- render stuff -----------------------------------------

    def render(self):
        if self.viewer is None:
            pygame.init()
            # Clock to control frame rate
            self.rt_clock = pygame.time.Clock()
            # set window
            self.viewer = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Gaze Fixation")
        # Fill the screen with white
        self.viewer.fill(GREY)
        # draw fov
        try: pygame.draw.polygon(self.viewer, WHITE, [self.pxl_coordinates((self.robot.pose.x,self.robot.pose.y)), self.pxl_coordinates(self.polar_point(self.robot.pose.phi+self.robot.sensor_angle/2, self.world_size*3)), self.pxl_coordinates(self.polar_point(self.robot.pose.phi-self.robot.sensor_angle/2, self.world_size*3))])
        except: self.viewer.fill(WHITE)

        # draw target distance
        pygame.draw.circle(self.viewer, RED, self.pxl_coordinates((self.target.pose.x,self.target.pose.y)), self.distance*self.scale, width=1)
        # draw target
        pygame.draw.circle(self.viewer, RED, self.pxl_coordinates((self.target.pose.x,self.target.pose.y)), self.robot.size/2*self.scale)

        # draw vision axis
        pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates((self.robot.pose.x,self.robot.pose.y)), self.pxl_coordinates(self.polar_point(self.robot.pose.phi,self.world_size*3)))

        # draw Agent
        pygame.draw.circle(self.viewer, BLUE, self.pxl_coordinates((self.robot.pose.x,self.robot.pose.y)), self.robot.size/2*self.scale)
        pygame.draw.polygon(self.viewer, BLUE, [self.pxl_coordinates(self.polar_point(self.robot.pose.phi+np.pi/2, self.robot.size/2.5)), self.pxl_coordinates(self.polar_point(self.robot.pose.phi-np.pi/2, self.robot.size/2.5)), self.pxl_coordinates(self.polar_point(self.robot.pose.phi, self.robot.size*0.7))])

        # draw obstacles
        for o in self.obstacles:
            pygame.draw.circle(self.viewer, BLACK, self.pxl_coordinates((o.pos.x,o.pos.y)), o.radius*self.scale)

        pygame.display.flip()
        self.rt_clock.tick(1/self.timestep)

    def polar_point(self, angle, distance):
        return self.robot.pose.x + distance * math.cos(angle), self.robot.pose.y + distance * math.sin(angle)
    
    def pxl_coordinates(self, xy):
        x_pxl = int(self.screen_size/2 + xy[0] * self.scale)
        y_pxl = int(self.screen_size/2 - xy[1] * self.scale)
        return (x_pxl, y_pxl)