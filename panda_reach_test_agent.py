import gym
import panda_gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#############################
# Load environment
env = gym.make("PandaReach-v1", render=True, reward_type="dense")

#############################
# Load trained model
model = PPO.load(
    "./rl-trained-agents/PandaReach_PPO_v1_model.zip",
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
