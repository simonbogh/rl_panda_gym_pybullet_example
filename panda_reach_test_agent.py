import gym
import panda_gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#############################
# Load environment
env = gym.make(
    "PandaReach-v2",
    render=True,
    reward_type="dense",  # "dense" or "sparse"
    control_type="ee",  # "ee" or "joints"
)

#############################
# Load trained model
model = PPO.load(
    "./rl-trained-agents/PandaReach_PPO_v2_ee_model.zip",
    print_system_info=True,
    device="auto",
)

#############################
# Reset environment and get first observation
obs = env.reset()

#############################
# Run trained agent in environment
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()  # Rendering in real-time (1x)
    if dones:
        obs = env.reset()
