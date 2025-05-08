"""
Q-Learning algorithm implemented by Caio Conti
"""

import numpy as np
import gymnasium as gym
import pygame
import pickle
from enum import Enum
import matplotlib.pyplot as plt
from utils import *
import os, sys
DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(DIR, '..'))) # Given that my vscode wasn't working with modules correctly
from envs.fixed_target import FixedTargetEnv
from envs.any_target import AnyTargetEnv

# SELECT PARAMETERS
ENVIRONMENT = 'any' # 'fixed' for fixed target or 'any' for target anywhere
RENDERING = False # True to render simulation for visualization
TRAINING = True # True to train, false to load result from TABLE_FILE
TABLE_FILE = 'q_table.pkl'
N_EPISODES = 100000 # Number of episodes

# Q learning algorithm
def q_learning(episodes, env_name, is_training = True, render=False, qtable_name = 'q_table.pkl'):
    print("Environment: ", env_name, "| Number of Episodes: ", episodes, "| Is training: ", is_training, "| Is rendering: ", render)

    # Hyperparameters
    learning_rate_a = 0.9
    discount_factor_g = 0.9

    # Environment selected
    if env_name == 'fixed':
        env = FixedTargetEnv(render_mode=render)
        # Prepare q-table (discretization)
        bins = np.array([10,10,10,10])
        n_states = np.prod(bins)
        n_actions = env.action_space.n

    elif env_name == 'any':
        env = AnyTargetEnv(render_mode=render)
        # Prepare q-table (discretization)
        agent_bins = np.array([10, 10, 10, 10])      # for 4D agent state: x, x_dot, y, y_dot
        target_bins = np.array([10, 10])          # for 2D target state: x_target, y_target    
        agent_bin_edges = create_bins(env.low_obs, env.high_obs, agent_bins)
        target_bin_edges = create_bins(np.array([0, 0]), np.array([env.x_size, env.y_size]), target_bins)
        n_states = np.prod(agent_bins) * np.prod(target_bins)
        n_actions = env.action_space.n

    endGraph = []

    # Prepare q-table    

    if(is_training):
        q_table = np.zeros((n_states,n_actions))
    else:
        f = open(qtable_name,'rb')
        q_table = pickle.load(f)
        f.close

    # Eps-greedy policy
    eps = 1
    eps_decay = 0.000012
    rng = np.random.default_rng()

    for episode in range(episodes):
        obs, info = env.reset()
        if env_name == "fixed":
            state = obs_to_index(obs, bins, env.low_obs, env.high_obs)
        elif env_name == "any":
            state = discretize_obs(obs, agent_bin_edges, target_bin_edges, agent_bins, target_bins)

        episode_over = False
        score = 0

        while not episode_over:
            if is_training and rng.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state,:])

            new_obs, reward, terminated, truncated, info = env.step(action)
            if env_name == "fixed":
                new_state = obs_to_index(new_obs, bins, env.low_obs, env.high_obs)
            elif env_name == "any":
                new_state = discretize_obs(new_obs, agent_bin_edges, target_bin_edges, agent_bins, target_bins)

            episode_over = terminated or truncated
            score+=reward
            if is_training:
                q_table[state,action] = q_table[state,action] + learning_rate_a*(
                    reward + discount_factor_g*np.max(q_table[new_state,:]) - q_table[state,action]
                )      
            state = new_state

        eps = max(eps - eps_decay,0)

        if eps == 0:
            learning_rate_a = 0.01

        if episode%100 == 0:
            print('Episode: ', episode, ', Score: ', score, ', Eps: ', eps)
        endGraph.append(score)

    if is_training:
        np.save('scores.npy', np.array(endGraph))
        f = open("q_table.pkl","wb")
        pickle.dump(q_table,f)
        f.close

    env.close()

if __name__ == "__main__":
    # Run
    q_learning(N_EPISODES,ENVIRONMENT,is_training=TRAINING,render=RENDERING,qtable_name=TABLE_FILE)

    # Loading and Ploting
    scores = np.load('scores.npy')
    window = 100
    rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid') # Rolling average over 100 episodes
    plt.figure(figsize=(10, 5))
    plt.plot(scores, alpha=0.3, label='Raw scores')
    plt.plot(np.arange(window-1, len(scores)), rolling_avg, label='Rolling Average (100)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Episode Score and Rolling Average')
    plt.legend()
    plt.grid(True)
    plt.show()