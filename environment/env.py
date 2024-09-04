import os
from typing import List
import gymnasium as gym
import numpy as np
import pygame
import vidmaker
import math

# =====================================================================================================

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 180, 0)
LIGHT_RED = (255, 200, 200)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)

SCREEN_SIZE = 900

# =====================================================================================================

class Robot:
    def __init__(self, pos, sensor_angle, max_vel, max_vel_rot, max_acc, max_acc_rot):
        self.start_pos: np.ndarray = pos
        self.pos: np.ndarray = pos
        self.orientation: float = 0.0
        self.vel: np.ndarray = np.array([0.0, 0.0], dtype=np.float64)
        self.vel_rot: float = 0.0
        self.sensor_angle: float = sensor_angle
        self.size: float = 0.5
        self.max_vel: float = max_vel
        self.max_vel_rot: float = max_vel_rot
        self.max_acc: float = max_acc
        self.max_acc_rot: float = max_acc_rot

    def reset(self):
        self.pos: np.ndarray = self.start_pos.copy()
        self.orientation: float = 0.0
        self.vel: np.ndarray = np.array([0.0, 0.0], dtype=np.float64)
        self.vel_rot: float = 0.0

class Obstacle:
    def __init__(self, radius = 1.0, pos = np.array([0.0, 0.0], dtype=np.float64)):
        self.radius: float = radius
        self.pos: np.ndarray = pos

class Target:
    def __init__(self, x=0.0, y=0.0):
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.array([0.0, 0.0], dtype=np.float64)

# =====================================================================================================

class Environment(gym.Env):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.episode_length = config["episode_length"]
        self.timestep: float = config["timestep"]
        self.num_steps: int = 0
        self.time = 0.0
        self.total_reward = 0.0

        self.observe_distance = config["observe_distance"]
        self.action_mode: int = config["action_mode"]
        self.action = np.array([0.0, 0.0, 0.0])

        self.target_distance: float = np.random.uniform(low=0.0,high=config["target_distance"])
        self.reward_margin: float = config["reward_margin"]
        self.penalty_margin: float = config["penalty_margin"]
        self.wall_collision: bool = config["wall_collision"]
        self.num_obstacles: int = config["num_obstacles"]
        self.use_obstacles: bool = config["use_obstacles"]

        # env dimensions
        self.world_size = config["world_size"]
        self.screen_size = SCREEN_SIZE
        self.scale = SCREEN_SIZE / self.world_size

        self.robot = Robot(np.array([-self.world_size/4, 0.0], dtype=np.float64), config["robot_sensor_angle"], config["robot_max_vel"], config["robot_max_vel_rot"], config["robot_max_acc"], config["robot_max_acc_rot"])
        self.obstacles: List[Obstacle] = []
        self.generate_target()
        self.generate_obstacles()

        if self.observe_distance:
            self.observation_space = gym.spaces.Box(
                low=np.array([self.target_distance, -self.robot.sensor_angle/2, -self.robot.max_vel_rot, -self.robot.max_vel, -self.robot.max_vel, -self.target_distance] + [-self.robot.sensor_angle/2, 0.0, -1.0] * self.num_obstacles),
                high=np.array([self.target_distance, np.pi, self.robot.max_vel_rot, self.robot.max_vel, self.robot.max_vel, np.inf] + [np.pi, 1.0, np.inf] * self.num_obstacles),
                shape=(6 + 3*self.num_obstacles,),
                dtype=np.float64
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([self.target_distance, -self.robot.sensor_angle/2, -self.robot.max_vel_rot, -self.robot.max_vel, -self.robot.max_vel] + [-self.robot.sensor_angle/2, 0.0] * self.num_obstacles),
                high=np.array([self.target_distance, np.pi, self.robot.max_vel_rot, self.robot.max_vel, self.robot.max_vel] + [np.pi, 1.0] * self.num_obstacles),
                shape=(5 + 2*self.num_obstacles,),
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
        self.render_relative_to_robot = 2
        self.record_video = False
        self.video_path = ""
        self.video = None

        self.reset(self.config["seed"])
    
    def step(self, action):
        self.num_steps += 1
        self.time += self.timestep
        if self.action_mode == 1:
            self.action = self.validate_action(action) # make sure acceleration vector is within bounds
        self.move_robot(action)
        self.move_target()
        obs, rew, done, trun, info = self.get_observation(), self.get_reward(), self.get_terminated(), False, self.get_info()
        self.total_reward += rew
        return obs, rew, done, trun, info
    
    def reset(self, seed=None, record_video=False, video_path = "", **kwargs):
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        if self.video is not None:
            os.makedirs(self.video_path, exist_ok=True)
            self.video.export(verbose=True)
            #self.video.compress(target_size = int(self.time/10 * 1024), new_file=True)
        self.record_video = record_video
        self.video_path = video_path
        self.time = 0.0
        self.num_steps = 0
        self.total_reward = 0.0
        self.action = np.array([0.0, 0.0, 0.0])
        self.robot.reset()
        self.collision = False
        self.generate_target()
        self.generate_obstacles()

        return self.get_observation(), self.get_info()
    
    def close(self):
        pygame.quit()
        self.screen = None
        
    def get_observation(self):
        angle = self.normalize_angle(np.arctan2(self.target.pos[1]-self.robot.pos[1], self.target.pos[0]-self.robot.pos[0]) - self.robot.orientation)
        if not (angle>-self.robot.sensor_angle/2 and angle<self.robot.sensor_angle/2):
            angle = np.pi
        obstacles = self.observe_obstacles()
        if self.observe_distance:
            return np.concatenate([np.array([self.target_distance, angle, self.robot.vel_rot, self.robot.vel[0], self.robot.vel[1], self.robot_target_distance()-self.target_distance]), obstacles])
        else:
            return np.concatenate([np.array([self.target_distance, angle, self.robot.vel_rot, self.robot.vel[0], self.robot.vel[1]]), obstacles])
    
    def get_reward(self):
        if self.collision:
            return - self.total_reward - 1.0
            #return - self.total_reward - self.episode_length
        reward = 0.0
        # reward for being close to target distance
        dist = self.robot_target_distance()
        if abs(dist-self.target_distance) < self.reward_margin:
            reward += 1.0 / (abs(dist-self.target_distance) + 1.0) * self.timestep
        # penalty for being close to obstacle
        if self.use_obstacles:
            for obstacle in self.obstacles:
                dist = np.linalg.norm(self.robot.pos-obstacle.pos)
                if abs(dist-obstacle.radius) < self.penalty_margin:
                    reward -= 1.0 / (abs(dist-obstacle.radius) + 1.0) * self.timestep
        # penalize energy waste
        reward -= np.linalg.norm(self.action[:2]) / self.robot.max_acc * self.timestep / 5
        reward -= abs(self.action[2]) / self.robot.max_acc_rot * self.timestep / 5
        return reward
    
    def get_terminated(self):
        return self.time > self.episode_length or self.collision
    
    def get_info(self):
        return {}
    
    # -------------------------------------- helpers -------------------------------------------

    def validate_action(self, action):
        acc = action[:2]
        acc_abs = np.linalg.norm(acc)
        if acc_abs > self.robot.max_acc:
            acc = (acc / acc_abs) * self.robot.max_acc
        acc_rot = action[2]
        if abs(acc_rot) > self.robot.max_acc_rot:
            acc_rot = acc_rot / abs(acc_rot) * self.robot.max_acc_rot
        return np.array([acc[0], acc[1], acc_rot])
    
    def move_robot(self, action):
        # set xy and angular accelerations
        if self.action_mode == 1:
            acc = np.array([action[0], action[1]])
            acc_rot = action[2]
        elif self.action_mode == 2:
            acc = np.array([(action[0]-1)*self.robot.max_acc, (action[1]-1)*self.robot.max_acc])
            acc_rot = (action[2] - 1) * self.robot.max_acc_rot
        self.robot.vel += acc * self.timestep                   # update robot velocity vector
        self.robot.vel_rot += acc_rot * self.timestep           # update rotational velocity
        self.limit_robot_velocity()

        # move robot
        self.robot.pos += (self.rotation_matrix(self.robot.orientation) @ self.robot.vel) * self.timestep
        del_orientation = self.robot.vel_rot * self.timestep
        self.robot.orientation += del_orientation
        self.robot.vel = self.rotation_matrix(-del_orientation) @ self.robot.vel

        # constrain orientation to range [-pi, pi]
        if self.robot.orientation < -np.pi: self.robot.orientation = self.robot.orientation + 2*np.pi
        elif self.robot.orientation > np.pi: self.robot.orientation = self.robot.orientation - 2*np.pi
        self.check_collision()

    def limit_robot_velocity(self):  
        vel = np.linalg.norm(self.robot.vel)                                            # compute absolute translational velocity
        if vel > self.robot.max_vel:                                                    # make sure translational velocity is within bounds
            self.robot.vel = self.robot.vel / vel * self.robot.max_vel
        if abs(self.robot.vel_rot) > self.robot.max_vel_rot:                            # make sure rotational velocity is within bounds
            self.robot.vel_rot = self.robot.vel_rot/abs(self.robot.vel_rot) * self.robot.max_vel_rot

    def move_target(self):
        #self.target.pos[0] += 1.0 * self.timestep
        pass
    
    def check_collision(self):
        if self.use_obstacles:
            for o in self.obstacles:
                if np.linalg.norm(np.array([o.pos[0]-self.robot.pos[0],o.pos[1]-self.robot.pos[1]])) < o.radius + self.robot.size / 2:
                    self.collision = True
                    return
        if self.wall_collision:
            if self.robot.pos[0] < -self.world_size / 2 + self.robot.size/2:
                self.robot.pos[0] = -self.world_size / 2 + self.robot.size/2
                self.collision = True
            elif self.robot.pos[0] > self.world_size / 2 - self.robot.size/2:
                self.robot.pos[0] = self.world_size / 2 - self.robot.size/2
                self.collision = True
            if self.robot.pos[1] < -self.world_size / 2 + self.robot.size/2:
                self.robot.pos[1] = -self.world_size / 2 + self.robot.size/2
                self.collision = True
            elif self.robot.pos[1] > self.world_size / 2 - self.robot.size/2:
                self.robot.pos[1] = self.world_size / 2 - self.robot.size/2
                self.collision = True

    def observe_obstacles(self):
        if self.use_obstacles:
            obstacles = []
            for obstacle in self.obstacles:
                obs = []
                # angle
                angle = self.normalize_angle(np.arctan2(obstacle.pos[1]-self.robot.pos[1], obstacle.pos[0]-self.robot.pos[0]) - self.robot.orientation)
                if not (angle>-self.robot.sensor_angle/2 and angle<self.robot.sensor_angle/2):
                    angle = np.pi
                    if self.observe_distance:
                        obstacles += [angle, 0.0, -1.0]
                    else:
                        obstacles += [angle, 0.0]
                    continue
                obs.append(angle)
                # coverage
                obs.append(self.calculate_circle_coverage(obstacle))
                # distance
                if self.observe_distance:
                    obs.append(np.linalg.norm(obstacle.pos-self.robot.pos)-obstacle.radius)
                obstacles += obs
        else:
            obs = [np.pi, 0.0]
            if self.observe_distance:
                obs.append(-1.0)
            obstacles = self.num_obstacles * obs
        return np.array(obstacles)  

    def robot_target_distance(self):
        return np.linalg.norm(self.target.pos-self.robot.pos)
    
    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def rotation_matrix(self, angle):
        return np.array(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )

    def calculate_circle_coverage(self, obstacle: Obstacle):
        angular_size = 2 * math.asin(min(1.0, obstacle.radius / np.linalg.norm(obstacle.pos-self.robot.pos)))
        coverage = angular_size / self.robot.sensor_angle
        return min(coverage, 1.0)
    
    def generate_target(self):
        distance = np.random.uniform(self.world_size / 4, self.world_size / 2)
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        self.target = Target(x, y)
    
    def generate_obstacles(self):
        self.obstacles = []
        target_distance = np.linalg.norm(self.target.pos)
        std_dev = target_distance / 3
        midpoint = (self.target.pos + self.robot.pos) / 2
        for _ in range(self.num_obstacles):
            while True:
                radius = max(np.random.random() * self.world_size / 6, self.world_size / 15)
                pos = np.random.normal(loc=midpoint, scale=std_dev, size=2)
                # Ensure the obstacle doesn't spawn too close to robot
                if np.linalg.norm(pos-self.robot.pos) > radius + self.penalty_margin + self.robot.size:
                    self.obstacles.append(Obstacle(radius, pos))
                    break

    # ----------------------------------- render stuff -----------------------------------------

    def render(self):
        if self.viewer is None:
            pygame.init()
            # Clock to control frame rate
            self.rt_clock = pygame.time.Clock()
            # set window
            self.viewer = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Gaze Fixation")
            if self.record_video:
                self.video = vidmaker.Video(self.video_path + "GazeFixation.mp4", fps=int(1/self.timestep), resolution=(self.screen_size,self.screen_size), late_export=True)
        # Fill the screen with white
        self.viewer.fill(GREY)
        # draw fov
        # TODO: Consider larger FOV's?
        if self.robot.sensor_angle < np.pi: pygame.draw.polygon(self.viewer, WHITE, [self.pxl_coordinates((self.robot.pos[0],self.robot.pos[1])), self.pxl_coordinates(self.polar_point(self.robot.orientation+self.robot.sensor_angle/2, self.world_size*3)), self.pxl_coordinates(self.polar_point(self.robot.orientation-self.robot.sensor_angle/2, self.world_size*3))])
        else: self.viewer.fill(WHITE)
        self.render_grid()
        # draw target distance margin
        self.transparent_circle(self.target.pos, self.target_distance+self.reward_margin, GREEN)
        # draw target distance
        pygame.draw.circle(self.viewer, DARK_GREEN, self.pxl_coordinates((self.target.pos[0],self.target.pos[1])), self.target_distance*self.scale, width=1)
        # draw target
        pygame.draw.circle(self.viewer, DARK_GREEN, self.pxl_coordinates((self.target.pos[0],self.target.pos[1])), self.robot.size/2*self.scale)
        # draw vision axis
        pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates((self.robot.pos[0],self.robot.pos[1])), self.pxl_coordinates(self.polar_point(self.robot.orientation,self.world_size*3)))
        # draw Agent
        pygame.draw.circle(self.viewer, BLUE, self.pxl_coordinates((self.robot.pos[0],self.robot.pos[1])), self.robot.size/2*self.scale)
        pygame.draw.polygon(self.viewer, BLUE, [self.pxl_coordinates(self.polar_point(self.robot.orientation+np.pi/2, self.robot.size/2.5)), self.pxl_coordinates(self.polar_point(self.robot.orientation-np.pi/2, self.robot.size/2.5)), self.pxl_coordinates(self.polar_point(self.robot.orientation, self.robot.size*0.7))])
        # draw obstacles
        if self.use_obstacles:
            for o in self.obstacles:
                pygame.draw.circle(self.viewer, BLACK, self.pxl_coordinates((o.pos[0],o.pos[1])), o.radius*self.scale)
                self.transparent_circle(o.pos, o.radius+self.penalty_margin, RED)
        self.display_info()
        if self.record_video:
            self.video.update(pygame.surfarray.pixels3d(self.viewer).swapaxes(0, 1), inverted=False)
        pygame.display.flip()
        self.rt_clock.tick(1/self.timestep)

    def transparent_circle(self, pos, radius, color):
        target_rect = pygame.Rect(self.pxl_coordinates(pos), (0, 0)).inflate((radius*2*self.scale, radius*2*self.scale))
        shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        pygame.draw.circle(shape_surf, (color[0],color[1],color[2],50), (radius*self.scale, radius*self.scale), radius*self.scale)
        self.viewer.blit(shape_surf, target_rect)

    def polar_point(self, angle, distance):
        return self.robot.pos[0] + distance * math.cos(angle), self.robot.pos[1] + distance * math.sin(angle)
    
    def display_info(self):
        font = pygame.font.Font(None, 24)
        clock_surface = font.render('Step:', True, BLACK)
        time_surface = font.render('Time:', True, BLACK)
        step_reward = font.render('Step reward:', True, BLACK)
        episode_reward = font.render('Total reward:', True, BLACK)
        clock_surface_val = font.render(f'{self.num_steps}', True, BLACK)
        time_surface_val = font.render('{0:.2f}'.format(self.time), True, BLACK)
        step_reward_val = font.render('{0:.4f}'.format(self.get_reward()), True, BLACK)
        episode_reward_val = font.render('{0:.4f}'.format(self.total_reward), True, BLACK)

        self.viewer.blit(clock_surface, (10, 5))
        self.viewer.blit(time_surface, (10, 30))
        self.viewer.blit(step_reward, (10, 55))
        self.viewer.blit(episode_reward, (10, 80))
        self.viewer.blit(clock_surface_val, (150, 5))
        self.viewer.blit(time_surface_val, (150, 30))
        self.viewer.blit(step_reward_val, (150, 55))
        self.viewer.blit(episode_reward_val, (150, 80))

    def render_grid(self):
        lines = []
        for x in range(int(self.robot.pos[0]-self.world_size/2), int(self.robot.pos[0]+self.world_size/2+1)):
            if x % 10 == 0:
                lines.append(((float(x),self.robot.pos[1]+self.world_size), (float(x),self.robot.pos[1]-self.world_size)))
        for y in range(int(self.robot.pos[1]-self.world_size/2), int(self.robot.pos[1]+self.world_size/2+1)):
            if y % 10 == 0:
                lines.append(((self.robot.pos[0]+self.world_size,float(y)), (self.robot.pos[0]-self.world_size,float(y))))
        for line in lines:
            pygame.draw.line(self.viewer, BLACK, self.pxl_coordinates(line[0]), self.pxl_coordinates(line[1]))

    def pxl_coordinates(self, xy):
        if self.render_relative_to_robot == 1:
            x_pxl = int(self.screen_size/2 + xy[0] * self.scale)
            y_pxl = int(self.screen_size/2 - xy[1] * self.scale)
        elif self.render_relative_to_robot == 2:
            x_pxl = int(self.screen_size/2 + (self.rotation_matrix(-self.robot.orientation + np.pi/2) @ (xy-self.robot.pos))[0] * self.scale)
            y_pxl = int(self.screen_size/2 - (self.rotation_matrix(-self.robot.orientation + np.pi/2) @ (xy-self.robot.pos))[1] * self.scale)
        return (x_pxl, y_pxl)