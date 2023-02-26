import gym
import panda_gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

"""
Train PPO agent in the PandaReach-v1 environment.

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
"""

#############################
# Create environment
env = gym.make(
    "PandaReach-v2",
    render=True,
    reward_type="dense",  # "dense" or "sparse"
    control_type="ee",  # "ee" or "joints"
)

#############################
# Set up policy and train agent

# Custom actor (pi) and value function (vf) networks
# of two layers of size 64 each with Relu activation function
policy_kwargs = dict(
    activation_fn=th.nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])]
)

#############################
# Set up PPO model
# https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.MultiInputPolicy
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="runs",
    batch_size=64,
    normalize_advantage=True,
    learning_rate=0.001,
    policy_kwargs=policy_kwargs,
)

#############################
# Train agent
model.learn(
    total_timesteps=100000,
    tb_log_name="PPO_lr0.001_hu64_64_bs64",
)

#############################
# Save trained model
model.save("PandaReach_PPO_v2_ee_model")
