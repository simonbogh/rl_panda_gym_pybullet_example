## Reinforcement Learning with the Franka Panda robot pybullet simulation
This is an example showing how to train a Reinforcement Learning agent using Pybullet, OpenAI gym, Panda Gym environments, and Stable-Baselines3.

**Panda pybullet simulation**
<!-- ![pybullet_simulation](docs/pybullet_simulation.png) -->

![pybullet_simulation_gif](docs/training.gif)

**Episode reward in Tensorboard**
![tensorboard](docs/tensorboard.png)

## Installation

### Create Python 3.8 virtual environment

```sh
conda create -n rl_panda_gym_py38 python=3.8
conda activate rl_panda_gym_py38
```

### Clone repo

```sh
git clone https://github.com/omtp_panda_gym.git
```

### Install python packages

We need the following Python , which are all defined in `requirements.txt`.

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
cd omtp_panda_gym
pip install -r requirements.txt
```