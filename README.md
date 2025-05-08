# RL-tests

This repository contains reinforcement learning (RL) experiments and exercises, with a focus on creating and solving custom environments for robotic learning.

## Custom Environments

We design new environments inspired by the [Gymnasium environment creation tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/). Unlike the tutorial, which focuses on discrete action and observation spaces, our environments operate in **continuous spaces**. To handle these, we adapt force-based dynamics similar to those used in the [CartPole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/).

## RL Algorithms

We apply two reinforcement learning methods to solve our environments:

- **Basic Q-Learning** (for envs fixed_target and any_target)
- **Deep Q-Learning (DQN)** using the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library, for all envs.

## Running Experiments

To test the Q-learning implementation:

1. Open `QLearning.py` in the `tests` folder.
2. Select one of the two environments you'd like to train within the script.
3. Execute.
