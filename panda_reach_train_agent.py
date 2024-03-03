import gymnasium as gym
import panda_gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime

"""
Train PPO agent in the PandaReach-v3 environment.

We use a dense reward function, which makes it easier to learn a
policy than with a sparse reward function. A sparse reward function
can also be used.

Reward functions:
    distance_threshold = 0.05
    d = distance(achieved_goal, desired_goal)  # Norm distance (Euclidean distance)

    Sparse: -(d > distance_threshold).astype(np.float32)
    Dense:  -d

Actions (control_type):
    "ee": end-effector x, y, z
    "joints": joint 1-7

Start tensorboard to track training progress:
    tensorboard --logdir=runs

Author: Simon Bøgh
"""

#############################
# Define Panda environment variables
env_name = "PandaReach-v3"
reward_type = "dense"  # "dense" or "sparse"
control_type = "joints"  # "ee" or "joints"
render_mode = "human"  # "human" or "rgb_array"

#############################
# Create environment
env = gym.make(
    env_name,
    render_mode=render_mode,  # "human" or "rgb_array"
    reward_type=reward_type,  # "dense" or "sparse"
    control_type=control_type,  # "ee" or "joints"
)

#############################
# Define variables for saving and logging
# Total training timestes
total_timesteps = 200000
# Learning rate for the PPO optimizer
learning_rate = 0.001
# Batch size for training the PPO model
batch_size = 64
# Number of hidden units for the policy network
pi_hidden_units = [64, 64]
# Number of hidden units for the value function network
vf_hidden_units = [64, 64]
# Current date as a string in the format "ddmmyyyy"
date_string = datetime.now().strftime("%d%m%Y")
# Name of the file to save the trained PPO model to
model_save_name = f"{env_name}_PPO_{control_type}_model_lr{learning_rate}_bs{batch_size}_pi{pi_hidden_units}_vf{vf_hidden_units}_{date_string}.zip"
# Name of the TensorBoard log directory for tracking training progress
tb_log_name = f"{env_name}_PPO_{control_type}_model_lr{learning_rate}_bs{batch_size}_pi{pi_hidden_units}_vf{vf_hidden_units}_{date_string}"

#############################
# Set up policy and train agent

# Custom actor (pi) and value function (vf) networks
# of two layers of size 64 each with Relu activation function
policy_kwargs = dict(
    activation_fn=th.nn.ReLU, net_arch=[dict(pi=pi_hidden_units, vf=vf_hidden_units)]
)

#############################
# Set up PPO model
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.MultiInputPolicy
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="runs",
    batch_size=batch_size,
    normalize_advantage=True,
    learning_rate=learning_rate,
    policy_kwargs=policy_kwargs,
)

#############################
# Train agent
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=tb_log_name,
)

#############################
# Save trained model
model.save(model_save_name)
