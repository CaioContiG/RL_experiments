"""
Agent tries to reach target location while being socially-aware.
In other words, minimize the discomfort felt by the people.
The agent induces a discomfort to each person, computed by Kirby: https://www.ri.cmu.edu/pub_files/2010/5/rk_thesis.pdf
Each person has a simple random walk, to simulate walking behaviour.
By Caio Conti
"""
import os, sys
DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(DIR, '..'))) # Given that my vscode wasn't working with modules correctly

from typing import Optional
import numpy as np
import time
import gymnasium as gym
import pygame
from enum import Enum
from tests.utils import *

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class SocialEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],"render_fps": 30}
    def __init__(self, render_mode=None, size: int = 10, n_people: int = 3):
        # Parameters
        self.size = size # Square environment
        self.x_size = self.size
        self.y_size = self.size
        self.vel_max = 1.5 # Max agent velocity
        self.tau = 0.05 # seconds between state updates
        self.force_mag = 2.0 # Magnitude of the force vector
        self.low_start = np.array([1, -0.1, 1, -0.1]) # Lower bound for the agent to spawn
        self.high_start = np.array([size-1, 0.1, size-1, 0.1]) # Upper bound for the agent to spawn
        
        # Set random people
        self.people = []
        for i in range(n_people):
            random_x = np.random.uniform(0, size)
            random_y = np.random.uniform(0, size)
            random_angle = np.random.uniform(-np.pi, np.pi)
            map_size = np.array([size,size])
            self.people.append(Person(random_x,random_y,random_angle,map_size))

        # Graphics Configs
        self.window_size = 512
        if render_mode: self.render_mode = "human"
        self.window = None
        self.clock = None   

        # We have 4 actions, corresponding to +x -x +y -y
        self.action_space = gym.spaces.Discrete(4)

        # Map action to force
        self.action_to_force = {
            Actions.RIGHT.value: [self.force_mag,  0.0],  # +x
            Actions.LEFT.value: [-self.force_mag,  0.0],  # -x
            Actions.UP.value: [0.0,             self.force_mag],  # +y
            Actions.DOWN.value: [0.0,            -self.force_mag],  # -y
        }

        '''
        Observation space:
        agent: x, x', y, y'
        target: x,y
        people_force: vector (x,y) of the total social force produce by people over the agent
        goal_direction: vector from agent to goal (dx, dy) = (x_agent,y_agent) - (x_target,y_target)
        distance2goal: distance from agent to goal
        '''
        self.low_obs = np.array([0.,-self.vel_max,0.,-self.vel_max],dtype=np.float32,)
        self.high_obs = np.array([self.x_size,self.vel_max,self.y_size,self.vel_max],dtype=np.float32,)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=self.low_obs, 
                    high=self.high_obs, 
                    dtype=np.float32), # Agent state x, x', y, y'
                "target": gym.spaces.Box(
                    low=np.array([0,0]), 
                    high=np.array([self.x_size,self.y_size]), 
                    dtype=np.float32
                    ),
                "people_force": gym.spaces.Box(
                    low=np.array([-5,-5]), # This is not a generalistic value, need to change
                    high=np.array([5,5]), # This is not a generalistic value, need to change
                    dtype=np.float32
                    ),
                "goal_direction": gym.spaces.Box(
                    low=np.array([-self.x_size, -self.y_size], dtype=np.float32),
                    high=np.array([self.x_size, self.y_size], dtype=np.float32),
                    dtype=np.float32
                    ),
                "distance_2_goal": gym.spaces.Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array([np.sqrt(self.x_size**2 + self.y_size**2)], dtype=np.float32),
                    dtype=np.float32
                )
            }
        )

        #self._agent_location = np.array([-1, -1], dtype=np.float32)
        #self._target_location = np.array([5, 1], dtype=np.float32)
        #self._target = np.array([5, 9], dtype=np.float32)
        # Set start state (x,x',y,y'),(xtarget,ytarget)
        #self.low_start = np.array([1, -0.1, 1, -0.1])
        #self.high_start = np.array([self.x_size-1, 0.1, self.y_size-1, 0.1])
        #self._agent = self.np_random.uniform(low=self.low_start,high=self.high_start , size=(4,))
        ##self._target = self.np_random.uniform(low = np.array([1,1]),high=np.array([self.y_size-1,self.x_size-1]))
        #self.prev_dist_to_goal = np.linalg.norm(self._agent_location - self._target_location)
        #self._people_force = np.array([0,0], dtype=np.float32)

    def step(self, action: np.ndarray):
        # Getting action and current observed state
        force_x, force_y = self.action_to_force[int(action)]   
        
        # Current state
        x, x_dot, y, y_dot = self._agent
        x_target, y_target = self._target

        """
        Compute new velocity and position using simple euler integration.
        Clip the velocity to stay between boundaries set.
        TODO: damping factor and see if it works.
        """
        x = x + self.tau * x_dot
        y = y + self.tau * y_dot
        x_dot = np.clip(x_dot + self.tau * force_x, -self.vel_max, self.vel_max)
        y_dot = np.clip(y_dot + self.tau * force_y, -self.vel_max, self.vel_max)

        # Update state
        self._agent = np.array((x, x_dot, y, y_dot), dtype=np.float32)
        self._agent_location = np.array([x,y]) # self._agent could be used, this variable is only for conveniance
        self._target_location = np.array([x_target,y_target]) # Not necessary, only for modifications where target is moving

        # Update person
        for person in self.people:
            person.random_walk()

        # Update force exert by people by summing all discomfort forces
        total_force = np.array([0.0, 0.0], dtype=np.float32)
        for person in self.people:
            force_magnitude = person.discomfort(self._agent_location)
            direction = self._agent_location - person.location
            distance = np.linalg.norm(direction)

            if distance > 0:
                unit_direction = direction / distance
            else:
                unit_direction = np.array([0.0, 0.0])  # Avoid division by zero

            force_vector = force_magnitude * unit_direction
            total_force += force_vector

        # Compute current distance to goal
        current_dist_to_goal = np.linalg.norm(self._agent_location - self._target_location,ord=2)

        # Store the result
        self._people_force = total_force

        ## Rewards ##
        reward = 0        
        reward_step_penalty = -1. # Step penalty to encourage faster solutions      
        reward_goal_dist = -current_dist_to_goal  # Goal penalty        
        reward_discomfort = -np.linalg.norm(total_force) # Discomfort penalty

        # Combine rewards with weighting
        reward = (
            1.0 * reward_goal_dist +
            8.0 * reward_discomfort +  # increase weight if the robot ignores discomfort
            reward_step_penalty
        )

        # Termination conditions
        terminated = False        

        # Terminate if outbounds
        terminated_out = bool(
            x < 0
            or x > self.size
            or y < 0
            or y > self.size
        )        
        
        # Terminate if reached goal        
        terminated_reached  = bool(
            np.linalg.norm(self._agent_location - self._target_location,ord=2) < 1.0
            )
        
        # Terminate if discomfort too high
        discomfort_people = [person.discomfort(self._agent_location) for person in self.people]        
        if any(d > 0.97 for d in discomfort_people):
            reward -=100
            terminated = True

        # Bonus for reaching goal
        if terminated_reached:
            reward += 100
            terminated = True

        # Penalty for going out of bounds
        if terminated_out:
            reward -= 100
            terminated = True

        # Neutral if reached steps max
        if  self.step_max <= 0:
            terminated = True

        # Decreasing steps
        self.step_max -= 1

        # Next frame
        if self.render_mode == "human":
            self.render()

        # return info
        return self._get_obs(), reward, terminated, False, {}
    
    def _get_obs(self):
        # Goal direction vector
        goal_direction = self._target_location - self._agent_location  # Vector from agent to goal

        # Distance to goal (Euclidean norm)
        distance_2_goal = np.linalg.norm(goal_direction)

        # Observation dictionary
        return {
            "agent": self._agent,
            "target": self._target,
            "people_force": self._people_force,
            "goal_direction": goal_direction.astype(np.float32),
            "distance_2_goal": np.array([distance_2_goal], dtype=np.float32)
        }

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        
        self._agent = self.np_random.uniform(low=self.low_start,high=self.high_start, size=(4,)).astype(np.float32)
        self._target = self.np_random.uniform(low = np.array([1,1]),high=np.array([self.x_size-1,self.y_size-1])).astype(np.float32)
        self._people_force = np.array([0,0], dtype=np.float32)
        self.state = [self._agent,self._target]
        self.step_max = 400
        x, x_dot, y, y_dot = self._agent
        x_target, y_target = self._target
        self._agent_location = np.array([x,y])
        self._target_location = np.array([x_target,y_target])
        self.prev_dist_to_goal = np.linalg.norm(self._agent_location - self._target_location)

        # Random people
        for person in self.people:
            random_x = np.random.uniform(0, self.x_size)
            random_y = np.random.uniform(0, self.y_size)
            random_theta = np.random.uniform(-np.pi, np.pi)
            person.set(random_x,random_y,random_theta)

        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), {}

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )       

        # Now we draw people
        for person in self.people:
            center = (person.location + 0.5) * pix_square_size
            angle = person.angle  # In radians, 0 = right, pi/2 = up, etc.
            size = pix_square_size / 3

            # Triangle tip and base points
            tip = center + size * np.array([np.cos(angle), np.sin(angle)])
            left = center + size * np.array([np.cos(angle + 2.5), np.sin(angle + 2.5)])
            right = center + size * np.array([np.cos(angle - 2.5), np.sin(angle - 2.5)])

            pygame.draw.polygon(canvas, (0, 255, 0), [tip, left, right])    

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

class Person():
    def __init__(self,x,y,theta,bounds):
        self.x = x
        self.y = y
        self.location = np.array([self.x,self.y])
        self.angle = theta
        self.x_bounds= (0,bounds[0])
        self.y_bounds= (0,bounds[1])

    def set(self,x,y,theta):
        self.x = x
        self.y = y
        self.location = np.array([self.x,self.y])
        self.angle = theta

    def discomfort(self,robot_location):
        x_robot, y_robot = robot_location
        return kirby(self.x,self.y,self.angle,x_robot,y_robot)
    
    def random_walk(self, step_size=0.02, angle_std=np.pi / 30,angle_change_prob = 0.3):
        # Store old position in case we revert
        old_x, old_y = self.x, self.y

        # Decide if we should change the angle based on the probability
        if np.random.rand() < angle_change_prob:
            delta_theta = np.random.normal(loc=0.0, scale=angle_std)
            self.angle += delta_theta

        # Small random change in angle (biased to continue forward)
        #delta_theta = np.random.normal(loc=0.0, scale=angle_std)
        #self.angle += delta_theta

        # Attempt move
        dx = step_size * np.cos(self.angle)
        dy = step_size * np.sin(self.angle)

        self.x += dx
        self.y += dy

        # Boundary check
        if not (self.x_bounds[0] <= self.x <= self.x_bounds[1]) or not (self.y_bounds[0] <= self.y <= self.y_bounds[1]):
            # Out of bounds — revert position and choose a new angle
            self.x, self.y = old_x, old_y
            self.angle += np.pi  # turn around

        self.location = np.array([self.x, self.y])