import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

env = gym.make('Pendulum-v0').unwrapped

# load optimized policy
model = Agent(state_size=3, action_size=1, random_seed=2)
model.actor_local.load_state_dict(torch.load('Pendulum.pth'))
model.actor_local.eval()

state = env.reset() 
# There are three observation: Cos(theta), sine(theta) that represents the angle of the pendulum and its angular velocity. 
# Theta is normalized between -pi and pi. Therefore, the lowest cost is -(π2 + 0.1*82 + 0.001*22) = -16.2736044, and the highest cost is 0
# The precise equation for reward is: -theta2 + 0.1*theta_dt2 +0.001*action2.
for t in range(1000):
    env.render()
    action = model.act(state, add_noise=False) # predict action. The action is a value between -2.0 and 2.0, representing the amount of left or right force on the pendulum
    state, reward, done, _ = env.step(action) # execute action
    print(t,f'{reward:.3f}')
    if done:
        break 
env.close()