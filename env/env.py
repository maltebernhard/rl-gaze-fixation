
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

WORLD_SIZE = 50.0
SCREEN_SIZE = 900

ROBOT_MAX_VEL = 3.0
ROBOT_MAX_VEL_PHI = 3.0
ROBOT_MAX_ACC = 3.0
ROBOT_MAX_ACC_PHI = 10.0

MAX_TIME = 30.0

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
    def __init__(self):
        self.pose: Pose = Pose()
        self.del_pose: Pose = Pose()
        self.sensor_angle = np.pi / 2
        self.size = 0.5
        self.max_vel = ROBOT_MAX_VEL
        self.max_vel_phi = ROBOT_MAX_VEL_PHI
        self.max_acc = ROBOT_MAX_ACC
        self.max_acc_phi = ROBOT_MAX_ACC_PHI

class Target:
    def __init__(self, x=0, y=0):
        self.pose = Pose(x,y)
        self.del_pose = Pose(0,0)

# =====================================================================================================

class Environment(gym.Env):
    def __init__(self, timestep, distance = 2.0):
        super().__init__()

        self.timestep = timestep
        self.time = 0.0

        self.distance = 2.0

        # env dimensions
        self.world_size = WORLD_SIZE
        self.screen_size = SCREEN_SIZE
        self.scale = SCREEN_SIZE / WORLD_SIZE

        self.robot = Robot()
        self.target = Target((np.random.random()-0.5)*self.world_size, (np.random.random()-0.5)*self.world_size)

        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(2),
                gym.spaces.Box(low=np.array([-self.robot.sensor_angle/2]), high=np.array([self.robot.sensor_angle/2]), shape=(1,))
                # include robot's dimensional velocities
                #gym.spaces.Box(low=np.array([-self.robot.sensor_angle/2,-self.robot.max_vel,-self.robot.max_vel,-self.robot.max_vel_phi]), high=np.array([self.robot.sensor_angle/2,self.robot.max_vel,self.robot.max_vel,self.robot.max_vel_phi]), shape=(4,))
            )
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-ROBOT_MAX_ACC, -ROBOT_MAX_ACC, -ROBOT_MAX_ACC_PHI]),
            high=np.array([ROBOT_MAX_ACC, ROBOT_MAX_ACC, ROBOT_MAX_ACC_PHI]),
            shape=(3,),
            dtype=np.float64
        )

        self.collision: bool = False

        # rendering window
        self.viewer = None
        metadata = {'render_modes': ['human'], 'render_fps': 1/timestep}
    
    def step(self, action):
        self.time += self.timestep
        action = self.validate_action(action) # make sure acceleration vector is within bounds
        self.move_robot(action)
        self.move_target()
        return self.get_observation(), self.get_reward(), self.get_terminated(), False, self.get_info()
    
    def reset(self, seed=None, **kwargs):
        self.time = 0.0
        self.robot = Robot()
        self.target = Target((np.random.random()-0.5)*self.world_size, (np.random.random()-0.5)*self.world_size)
        self.collision = False
        return self.get_observation(), self.get_info()
    
    def close(self):
        pygame.quit()
        self.screen = None
        
    def get_observation(self):
        angle = np.arctan2(self.target.pose.y-self.robot.pose.y, self.target.pose.x-self.robot.pose.x) - self.robot.pose.phi
        return (int(angle>-self.robot.sensor_angle/2 and angle<self.robot.sensor_angle/2), np.array([self.normalize_angle(angle)]))
        # include robot velocity into state
        #return (int(angle>-self.robot.sensor_angle/2 and angle<self.robot.sensor_angle/2), np.array([angle, self.robot.del_pose.x, self.robot.del_pose.y, self.robot.del_pose.phi]))
    
    def get_reward(self):
        # if self.collision:
        #     return -1000000
        dist = self.target_distance()
        if abs(dist-self.distance) < 1.0:
            return 1.0 / (abs(dist-self.distance) + 1.0)
        return 0.0
    
    def get_terminated(self):
        return self.time > MAX_TIME or self.collision
    
    def get_info(self):
        return {}
    
    # -------------------------------------- helpers -------------------------------------------

    def validate_action(self, action):
        translative_acc = action[:2]
        translative_length = np.linalg.norm(translative_acc)
        if translative_length > ROBOT_MAX_ACC:
            translative_acc = (translative_acc / translative_length) * ROBOT_MAX_ACC
        return np.array([translative_acc[0], translative_acc[1], action[2]])
    
    def move_robot(self, action):
        xy_acc = np.array([[np.cos(self.robot.pose.phi), -np.sin(self.robot.pose.phi)], [np.sin(self.robot.pose.phi), np.cos(self.robot.pose.phi)]]) @ np.array([action[0], action[1]])
        self.robot.del_pose += np.concatenate([xy_acc, np.array([action[2]])]) * self.timestep                           # apply acceleration to robot's velocity
        vel_vec = np.array([self.robot.del_pose.x, self.robot.del_pose.y])
        vel = np.linalg.norm(vel_vec)                                           # compute absolute velocity
        if vel > ROBOT_MAX_VEL:                                                 # make sure absolute velocity is within bounds
            vel_vec = vel_vec / vel * ROBOT_MAX_VEL
        del_phi = self.robot.del_pose.phi
        if abs(del_phi) > ROBOT_MAX_VEL_PHI:
            del_phi = np.sign(del_phi) * ROBOT_MAX_VEL_PHI
        self.robot.del_pose = Pose(vel_vec[0], vel_vec[1], del_phi)
        # move robot
        self.robot.pose += self.robot.del_pose * self.timestep
        # constrain phi to range [-pi, pi] 
        if self.robot.pose.phi < -np.pi: self.robot.pose.phi = self.robot.pose.phi + 2*np.pi
        elif self.robot.pose.phi > np.pi: self.robot.pose.phi = self.robot.pose.phi - 2*np.pi
        self.check_collision()

    def move_target(self):
        # TODO: implement
        pass
    
    def check_collision(self):
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

        pygame.display.flip()
        self.rt_clock.tick(1/self.timestep)

    def polar_point(self, angle, distance):
        return self.robot.pose.x + distance * math.cos(angle), self.robot.pose.y + distance * math.sin(angle)
    
    def pxl_coordinates(self, xy):
        x_pxl = int(self.screen_size/2 + xy[0] * self.scale)
        y_pxl = int(self.screen_size/2 - xy[1] * self.scale)
        return (x_pxl, y_pxl)