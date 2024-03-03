import gymnasium as gym
import panda_gym
import torch as th
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from datetime import datetime
from stable_baselines3.common.vec_env import VecNormalize

"""
Train TQC agent in the PandaReach-v3 environment.

TQC is able to train on sparse rewards, and is able to learn
a policy that is able to reach the goal.

Control type and training time with a sparse reward function:
 - "ee": 26.000 timesteps (20 min)
 - "joints": 110.000 timesteps (1hr 30 min)

 Author: Simon Bøgh
"""

#############################
# Define Panda environment variables
env_name = "PandaReach-v3"
reward_type = "sparse"  # "dense" or "sparse"
control_type = "joints"  # "ee" or "joints"
render_mode = "human"

#############################
# Create environment
env = gym.make(
    env_name,
    render_mode=render_mode,
    reward_type=reward_type,  # "dense" or "sparse"
    control_type=control_type,  # "ee" or "joints"
)

#############################
# Define custom policy
policy_kwargs = dict(
    n_critics=2,
    n_quantiles=25,
    net_arch=[256, 256],
)

#############################
# Set up TQC model
model = TQC(
    "MultiInputPolicy",
    env,
    buffer_size=1000000,
    top_quantiles_to_drop_per_net=2,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log="runs",
    batch_size=256,
    learning_rate=0.0003,
    target_update_interval=1,
    learning_starts=100,
)

#############################
# Train model
model.learn(total_timesteps=200000, log_interval=4, progress_bar=True)

#############################
# Save model
model.save("tqc_franka")
