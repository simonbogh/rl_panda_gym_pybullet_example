import gym
import panda_gym


def run_random_agent():
    #############################
    # Import gym environment
    env = gym.make("PandaReach-v1", render=True, reward_type="dense")

    #############################
    # Print action and observation space details
    print("==========================================")
    print("Action space:")
    print(env.action_space)
    print("Observation space:")
    print(env.observation_space)
    print("==========================================")

    #############################
    # Reset environment and get first observation
    obs = env.reset()
    done = False
    while True:
        # Take random action
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        env.render()  # Make the rendering real-time (1x)

    env.close()


if __name__ == "__main__":
    run_random_agent()
