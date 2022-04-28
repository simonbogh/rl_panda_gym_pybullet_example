## Reinforcement Learning with the Franka Panda robot pybullet simulation
This is an example showing how to train a Reinforcement Learning agent using:
* [PyBullet](https://github.com/bulletphysics/bullet3) physics simulation
* [OpenAI gym](https://github.com/openai/gym) RL API
* [panda-gym](https://github.com/qgallouedec/panda-gym) robot environments in PyBullet
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) RL algorithms library

**Panda pybullet simulation**
<!-- ![pybullet_simulation](docs/pybullet_simulation.png) -->

![pybullet_simulation_gif](docs/training.gif)

**Episode reward in Tensorboard**
![tensorboard](docs/tensorboard.png)

## Reinforcement Learning environment

### Actions
Action space:

The `PandaReach-v2` environment action space is composed of the gripper movement, three coordinates **Box(3)**, one for each axis of movement x, y, z, or joint movement, seven joints **Box(7)**.

```
Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)

Box([-1. -1. -1. -1. -1. -1. -1.], [1. 1. 1. 1. 1. 1. 1.], (7,), float32)
```

The action space can be selected by setting `control_type` when creating the environment.

```python
env = gym.make(
    "PandaReach-v2",
    render=True,
    reward_type="dense",  # "dense" or "sparse"
    control_type="ee",  # "ee" or "joints"
)
```

### Observation
Observation space:

The `PandaReach-v2` environment has the following observations, where *observation* is the position and speed of the gripper **Box(6)**.

```
Dict(
    achieved_goal:
        Box([-10. -10. -10.], [10. 10. 10.], (3,), float32),
    desired_goal:
        Box([-10. -10. -10.], [10. 10. 10.], (3,), float32),
    observation:
        Box([-10. -10. -10. -10. -10. -10.], [10. 10. 10. 10. 10. 10.], (6,), float32)
    )
```

### Reward
The `PandaReach-v2` environment comes with both `sparse` and `dense` reward functions. Default is the sparse reward function, which returns 0 or -1 if the *desired goal* was reached within some tolerance. The dense reward function is the negative of the distance *d* between the *desired goal* and the *achieved goal*.

```python
distance_threshold = 0.05  # Distance threshold in meters
d = distance(achieved_goal, desired_goal)  # Norm distance (Euclidean distance)

if self.reward_type == "sparse":
    return -(d > self.distance_threshold).astype(np.float32)
else:
    return -d
```

## Installation

### Create Python 3.8 virtual environment
The python virtual environment is set up with `Python 3.8`. Python 3.7 and up should also work, but has not been tested.

In the following we use Anaconda to set up the virtual environment.

```sh
$ conda create -n rl_panda_gym_py38 python=3.8
$ conda activate rl_panda_gym_py38
```

### Clone repo

```sh
$ git clone https://github.com/simonbogh/rl_panda_gym_pybullet_example.git
```

### Install python packages

We need the following Python packages, which are all defined in `requirements.txt`.

* black==22.3.0
* gym==0.21.0
* panda-gym==2.0.0
* pybullet==3.2.4
* pyglet==1.5.23
* sb3-contrib==1.5.0
* scikit-learn==1.0.2
* tensorboard==2.8.0
* torch==1.11.0
* wandb==0.12.15

```
$ cd rl_panda_gym_pybullet_example
$ pip install -r requirements.txt
```

## Random agent
The following example tests the installation of `panda_gym` and `pybullet`. It runs the `PandaReach-v2` pybullet environment and sends random actions to the robot.

No Reinforcement Learning agent is trained. Check the next section for how to train an agent using Stable-Baselines3.

```python
import gym
import panda_gym  # Import panda_gym to register the Panda pybullet environments


def run_random_agent():
    # Create gym training environment
    env = gym.make(
        "PandaReach-v2",
        render=True,
        reward_type="dense",  # "dense" or "sparse"
        control_type="ee",  # "ee" or "joints"
    )
    print("Action space:")
    print(env.action_space)
    print("Observation space:")
    print(env.observation_space)

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
```

## Train PPO agent with Stable-Baselines3
Minimal example showing how to train a PPO agent for the `PandaReach-v2` task using Stable-Baselines3.

```python
import gym
import panda_gym
from stable_baselines3 import PPO

# Create gym training environment
env = gym.make(
    "PandaReach-v2",
    render=True,
    reward_type="dense",  # "dense" or "sparse"
    control_type="ee",  # "ee" or "joints"
)

# Set up PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="runs",
    learning_rate=0.001,
)

# Train agent
model.learn(total_timesteps=200000)

# Save trained model
model.save("PandaReach_v2_model")
```

The following python script includes more hyper-parameters and settings that can be adapted when training the agent.

```sh
$ python panda_reach_train_agent.py
```

![pybullet_simulation_gif](docs/training.gif)

## Test PPO agent
A trained agent/policy can be tested by loading the model saved during training. A trained model can be found in the folder `rl-trained-agents/`.

`panda_reach_test_agent.py` loads a saved model and runs the pybullet simulation using that model as the policy for the agent.

```sh
$ python panda_reach_test_agent.py
```

![pybullet_simulation_gif](docs/test_agent.gif)

## Using tensorboard to track training
Tensorboard log files are saved to the `runs/` folder. Make sure to set `tensorboard_log="runs"` for the PPO model before training. When training is running, run the following in your terminal to start tensorboard.

```sh
$ tensorboard --logdir runs
```

Logs can be viewed in a web browser at http://localhost:6006/

![tensorboard2](docs/tensorboard2.png)

## Using wandb to track training
Weights & Biases (wandb) can be used instead of tensorboard to track the training process. Several runs can be compared and you can make reports for the experiments.

https://docs.wandb.ai/guides/integrations/other/stable-baselines-3
![wandb](docs/wandb.png)
