# Reinforcement Learning Experiments

This repository contains reinforcement learning (RL) experiments and exercises, with a focus on creating and solving custom environments tailored for robotic learning and social navigation.

## Custom Environments

We design custom environments inspired by the [Gymnasium environment creation tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/), extending its ideas to support **continuous** state and action spaces. Force-based dynamics, similar to those found in the [CartPole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/), are used for agent control.

### Environments

1. **Fixed Target (`fix_target.py`)**  
   The agent must navigate to a **static target** location. The goal remains unchanged.

2. **Any Target (`any_target.py`)**  
   The agent must reach a **randomized target** location that changes each episode.

3. **Socially-Aware Environment (`social_env.py`)**  
   The agent operates in a space with **moving pedestrians**, each with a **personal discomfort function** based on the work of [Kirby (2010)](https://www.ri.cmu.edu/pub_files/2010/5/rk_thesis.pdf). This function models how uncomfortable a person feels as the robot approaches, depending on proximity and orientation.

   The agent must:
   - Reach the randomized target,
   - Avoid collisions with people,
   - **Minimize social discomfort**, navigating in a human-aware way by learning socially compliant paths.


## RL Algorithms

We compare two reinforcement learning methods:

- **Basic Q-Learning**
  - Suitable for smaller, discrete state/action problems.
  - Solves: `Fixed Target` and `Any Target`.

- **Deep Q-Network (DQN)** — powered by [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
  - Solves: **All environments**, including the socially-aware one.

## Running Experiments

### Q-Learning
To apply Q-learning to the Fixed Target and Any Target environments, discretization is used to convert the continuous state space into a finite set of states for the Q-table. For the Fixed Target environment, the algorithm was tested on a 10×10 map. In the Any Target environment, it works only on smaller maps, such as 5×5. However, results may not be optimal, particularly in the Any Target case—since better performance would require finer discretization and more extensive training as Q-learning is not well-suited for high-dimensional/continuous problems.

1. Open the `tests/QLearning.py` script.
2. Choose either `fixed_target` or `any_target` in the script.
3. Run the file.

### DQN

We use a Deep Q-Network (DQN) implementation based on the [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) library to solve all environments, including the socially-aware one. Unlike basic Q-learning, DQN uses a neural network to approximate the Q-function, making it more suitable for environments with continuous or high-dimensional state spaces. It consistently achieves better results than Q-learning, especially in more complex tasks like socially-aware navigation.

#### Running DQN Training

Use the `DQN.py` script inside the `tests/` folder:

1. Open the `tests/DQN.py` script.
2. Choose either `fixed`, `any` or `social` in the script.
3. Run the file.


---

### Configurable Parameters

The scripts `QLearning.py` and `DQN.py` include user-defined parameters that control the environment, rendering, and other settings. You can modify these at the top of each script.

- **`ENVIRONMENT`**: Choose which environment to train/test on.  
  Options: `"fixed"`, `"any"`, or `"social"`

- **`RENDERING`**: If set to `True`, the simulation will visually render each step using pygame (useful for debugging or demonstrations).

- **`TRAINING`**: Set to `True` to train a new model. If `False`, the script will load a pre-trained model for testing.

- **`N_TIMESTEPS`**: The number of timesteps used for training the agent. (DQN ONLY)

- **`N_TESTS`**: The number of test episodes to evaluate the trained model (only used when `TRAINING` is set to `False`). (DQN ONLY)

- **`TABLE_FILE`**: The filename used to save/load the Q-table when training is disabled. (Q-Learning ONLY)

- **`N_EPISODES`**: The total number of training episodes to run. (Q-Learning ONLY)
