import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import gymnasium as gym
from rlss_envs import ServerlessEnv 
from gymnasium import spaces

env = ServerlessEnv()
raw_action = env.action_space.sample()
print(env._container_matrix)
env.reset()
print(raw_action)
next_state, reward, done, truncated, _ = env.step(raw_action)
print(env._container_matrix)
print(reward)
print(env.truncated_reason)