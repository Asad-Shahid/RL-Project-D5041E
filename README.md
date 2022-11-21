# RL Project - Applied case studies of Machine Learning (C-D5041E) 

This  repo contains a python toolkit for learning a grasping task with Franka Emika Panda Robot. The robot can be trained to grasp and lift the cube using a RL algorithm [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347). It is powered by [MuJoCo physics engine](http://www.mujoco.org/) 

# Installation

In order to use this toolkit, you first have to install [MuJoCo 200](https://www.roboti.us/download.html) simulation engine. [mujoco-py](https://github.com/openai/mujoco-py) by Open AI allows to use MuJoCo through python interface.
The installation requires python 3.6 or higher. It is recommended to install all the required packages under a conda virtual environment. 

After installing mujoco, the new conda environment can be created from ```environment.yml``` file using:

```
conda env create -f environment.yml
```
This will create a new conda environment named ```rl-project``` will all the required packages and dependencies.

To test the installation, activate the environment ```rl-project``` using:
```
conda activate rl-project
```

Then navigate to the ```panda``` directory, and run the following command: 

```
python -m rl.main
```
This should load the pre-trained policy and run 10 evaluation trials of the task.

# The Task
 * The main task of this project revolves around designing a reward function such that the robot can approach the cube placed on a table, grasp it and then lift it off. The idea of providing intermediate rewards to RL agent (as opposed to sparse reward for task completion) is known as [reward shaping](https://arxiv.org/pdf/1704.03073.pdf). It is often considered as a part of the learning system and is a key for RL agents to succeed at continous control tasks. In order to design the reward, you only have to modify the file [environment/panda_grasping.py](panda/environments/panda_grasping.py)

 * After designing the reward function, you are also encouraged to test different hyperparameters settings in [config/__init__.py](panda/config/__init__.py) and see the effect on training performance.


## References
This toolit is mainly developed based on [Surreal Robotics Suite](https://github.com/StanfordVL/robosuite) and the Reinforcement learning part is referenced from
[this repo](https://github.com/clvrai/furniture)
