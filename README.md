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

??? (ee_position, ee_velocity, [fingers_width])

```
Box([-1. -1. -1.], [1. 1. 1.], (3,), float32)
```

### Observation
Observation space:

??? observation = robot state ???

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
```python
if self.reward_type == "sparse":
    return -(d > self.distance_threshold).astype(np.float32)
else:
    return -d
```
Sparse

Dense

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
$ git clone https://github.com/omtp_panda_gym.git
```

### Install python packages

We need the following Python packages, which are all defined in `requirements.txt`.

* gym==0.21.0
* panda-gym==1.1.1
* pybullet==3.2.2
* pyglet==1.5.23
* sb3-contrib==1.5.0
* scikit-learn==1.0.2
* tensorboard==2.8.0
* torch==1.11.0
* wandb


```
$ cd omtp_panda_gym
$ pip install -r requirements.txt
```

## Random agent
The following example tests the installation of `panda_gym` and `pybullet`. It runs the `PandaReach-v1` pybullet environment and sends random actions to the robot.

No Reinforcement Learning agent is trained. Check the next section for how to train an agent using Stable-Baselines3.

```python
import gym
# Import panda_gym to register the Panda pybullet environments
import panda_gym


def run_random_agent():
    # Create gym training environment
    env = gym.make("PandaReach-v1",
                   render=True,
                   reward_type="dense")

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
Minimal example showing how to train a PPO agent for the `PandaReach-v1` task using Stable-Baselines3.

```python
import gym
import panda_gym
from stable_baselines3 import PPO

# Create gym training environment
env = gym.make("PandaReach-v1",
               render=True,
               reward_type="dense")

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
model.save("PandaReach_v1_model")
```

The following python script includes more hyper-parameters and settings that can be adapted when training the agent.

```sh
$ python panda_reach_train_agent.py
```

![pybullet_simulation_gif](docs/training.gif)

## Test PPO agent

```sh
$ python panda_reach_test_agent.py
```

![pybullet_simulation_gif](docs/test_agent.gif)

## Using tensorboard to track training
Tensorboard log files are saved to the `runs/` folder.

```sh
$ tensorboard --logdir runs
```

Logs can be viewed in a web browser at http://localhost:6006/

![tensorboard2](docs/tensorboard2.png)

## Using wandb to track training
Weights & Biases (wandb) can be used instead of tensorboard to track the training process. Several runs can be compared and you can make reports for the experiments.

https://docs.wandb.ai/guides/integrations/other/stable-baselines-3
![wandb](docs/wandb.png)
