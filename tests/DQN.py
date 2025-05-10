"""
DQN from stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#stable_baselines3.dqn.MlpPolicy
Configured to solve the environments.
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
import pickle
from enum import Enum
import matplotlib.pyplot as plt
from utils import *

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from envs.fixed_target import FixedTargetEnv
from envs.any_target import AnyTargetEnv
from envs.dynamic_people import SocialEnv

# SELECT PARAMETERS
ENVIRONMENT = "social"     # 'fixed', 'any' or 'social'
RENDERING = False       # True to render simulation for visualization
TRAINING = True         # True to train, false to load model for testing
N_TIMESTEPS = 300000    # Number of timesteps
N_TESTS = 100           # Number of episodes to test, when testing

# Environments
env_classes = {
    "fixed": FixedTargetEnv,
    "any": AnyTargetEnv,
    "social": SocialEnv,
}

def train_dqn(env,n_steps):
    # Monitor environment
    env = Monitor(env, filename="./logs/training/")
    
    # Choose policy based on env:
    if ENVIRONMENT == "fixed":
        choosed_policy = "MlpPolicy"
    else:
        choosed_policy = "MultiInputPolicy"
    
    # Initialize model
    print("Initializing Model...")    
    model = DQN(
        choosed_policy, 
        env, 
        verbose=1, 
        buffer_size=10000, 
        learning_starts=1000, 
        batch_size=32, 
        gamma=0.99,
        learning_rate=12e-5,
        exploration_fraction=0.5,
        exploration_final_eps=0.1
    )

    # Train
    model.learn(total_timesteps=n_steps)

    # Saving
    model.save("dqn_model")

def test_dqn(env,n_tests,model_name="dqn_model"):
    # Load the trained model
    model = DQN.load(model_name, env=env)

    # Monitor env
    env = Monitor(env, filename="./logs/testing/")

    # Run n_tests episodes using the trained policy
    for episode in range(n_tests):
        seed = int(time.time()) # "random" seed as super().reset(seed=seed) is not working as intended
        obs, _verbose = env.reset(seed=seed)
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _verbose, info = env.step(action)
            episode_reward += reward            
            env.render()

        print(f"Episode {episode + 1} reward: {episode_reward}")

def plot_results(option):
    # Plot results
    results = load_results("./logs/"+ option + "/")
    x, y = ts2xy(results, 'timesteps')

    #Rolling average over 100 episodes
    window = 100
    rolling_avg = np.convolve(y, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.plot(y, alpha=0.3, label='Raw scores')
    plt.plot(np.arange(window-1, len(y)), rolling_avg, label='Rolling Average (100)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Episode Score and Rolling Average')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    results = load_results('./logs/training')
    x, y = ts2xy(results, 'timesteps')
    plot_results("training")
    if ENVIRONMENT in env_classes:
        env = env_classes[ENVIRONMENT](render_mode=RENDERING)
    else:
        raise ValueError(f"Unknown ENVIRONMENT: {ENVIRONMENT}")

    check_env(env)

    if TRAINING:
        train_dqn(env,n_steps=N_TIMESTEPS)
    else:
        test_dqn(env,n_tests=N_TESTS)

    # Plot "training" or "testing" results
    plot_results("training")