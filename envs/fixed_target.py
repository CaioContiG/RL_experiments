"""
Continuous 2D agent with fixed target implemented by Caio Conti
Based on https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
And https://gymnasium.farama.org/environments/classic_control/cart_pole/
"""
from typing import Optional
import numpy as np
import time
import gymnasium as gym
import pygame
import pickle
from enum import Enum
import matplotlib.pyplot as plt

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class FixedTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, render_mode=None, size: int = 10):        
        
        # Parameters
        self.size = size # Square environment
        self.x_size = self.size
        self.y_size = self.size
        self.vel_max = 1.5 # Max agent velocity
        self.tau = 0.05 # seconds between state updates
        self.force_mag = 2.0 # Magnitude of the force vector
        self._target_location = np.array([5.,5.]) # Choose fixed target location
        self.low_start = np.array([1., -0.1, 1., -0.1]) # Lower bound for the agent to spawn
        self.high_start = np.array([size-1, 0.1, size-1, 0.1]) # Upper bound for the agent to spawn

        # Graphics Configs
        self.window_size = 512  # The size of the PyGame window
        #assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode: self.render_mode = "human"
        self.window = None
        self.clock = None        

        # We have 4 actions, corresponding to +x -x +y -y
        self.action_space = gym.spaces.Discrete(4)        

        # Maps abstract actions to force applied to the agent
        self.action_to_force = {
            Actions.RIGHT.value:    [self.force_mag,    0.0],  # +x
            Actions.LEFT.value:     [-self.force_mag,   0.0],  # -x
            Actions.UP.value:       [0.0,   self.force_mag],  # +y
            Actions.DOWN.value:     [0.0,   -self.force_mag],  # -y
        }

        # Set observation space x, x', y, y'
        self.low_obs = np.array([0,-self.vel_max,0,-self.vel_max,],dtype=np.float32,)
        self.high_obs = np.array([self.x_size,self.vel_max,self.y_size,self.vel_max,],dtype=np.float32)
        self.observation_space = gym.spaces.Box(self.low_obs, self.high_obs, dtype=np.float32)

        # Initialize parameters. Not necessary since they are being initialized at reset()
        #self.step_max = 300 # Max number of steps allowed
        #self.state = self.np_random.uniform(low=self.low_start,high=self.high_start , size=(4,))
        #self._agent_location = np.array([self.state[0],self.state[2]])
        #self._target_location = np.array([5.,5.])

    def step(self, action):
        
        # Getting action and current observed state
        force_x, force_y = self.action_to_force[int(action)]   
        x, x_dot, y, y_dot = self.state

        """
        Compute new velocity and position using simple euler integration.
        Clip the velocity to stay between boundaries set.
        TODO: damping factor and see if it works.
        """
        x = x + self.tau * x_dot
        y = y + self.tau * y_dot
        x_dot = np.clip(x_dot + self.tau * force_x, -self.vel_max, self.vel_max)
        y_dot = np.clip(y_dot + self.tau * force_y, -self.vel_max, self.vel_max)

        # Update state and agent with new position and velocity
        self.state = np.array((x, x_dot, y, y_dot), dtype=np.float32)
        self._agent_location = np.array([self.state[0],self.state[2]])

        # Initialize terminated flag
        terminated = False

        # Terminate if outbounds
        terminated_out = bool(
            x < 0
            or x > self.size
            or y < 0
            or y > self.size
        )

        # Terminate if near target
        terminated_reached  = bool(
            np.linalg.norm(self._agent_location - self._target_location,ord=2) < 1.2
            )

        """
        Compute rewards. Negative in relation to the distance from target.
        Other reward type would be getting positive rewards if the agent gets closer
        and negative rewards if agent gets farther.
        If target reach goal, gets positive reward of 1.
        """
        reward = -np.linalg.norm(self._agent_location - self._target_location,ord=2)

        if terminated_reached:
            reward = 1.

        if terminated_out or terminated_reached or self.step_max <= 0:
            terminated = True

        # Decreasing steps
        self.step_max -= 1

        # Render next frame
        if self.render_mode == "human":
            self.render()

        # Return information
        return self.state, reward, terminated, False, {}

    def reset(self, seed = None, options=None):
        super().reset(seed=seed)
        
        self.state = self.np_random.uniform(low=self.low_start, high=self.high_start, size=(4,)).astype(np.float32) # Create agent with random state        
        self._agent_location = np.array([self.state[0],self.state[2]]) # Update agent location        
        self.step_max = 200 # Set max number of steps

        # Render frame
        if self.render_mode == "human":
            self.render()
        
        return self.state, {}

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