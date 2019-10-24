#!/usr/bin/env python
# coding: utf-8

# In[11]:


import math, random

import gym
import numpy as np

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.autograd as autograd
# import torch.nn.functional as F

# from tensorboardX import SummaryWriter 

import matplotlib.pyplot as plt

import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '1' 


# In[12]:


# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '4x4', 'is_slippery': False},
#     max_episode_steps=100,
# #     reward_threshold=0.78, # optimum = .8196
# )


# In[13]:


env_id = 'FrozenLakeNotSlippery-v0'
env = gym.make(env_id)

"""
use gym (openAI)
https://blog.techbridge.cc/2017/11/04/openai-gym-intro-and-q-learning/

"""
print(env.observation_space.n)
print(env.action_space.n)


# # epsilon greedy

# In[14]:


epsilon_start = 1.
epsilon_final = 0.01
epsilon_decay = 3000.

def epsilon_by_frame(frame_idx):
    """
    your design
    """
    epsilon = max(math.exp(-(1/epsilon_decay)*frame_idx), epsilon_final)
#     epsilon = math.exp(-(1/epsilon_decay)*frame_idx)

    return epsilon


# In[15]:


plt.plot([epsilon_by_frame(i) for i in range(100000)])


# # Act

# In[16]:


def act(state, epsilon):
    if random.random() > epsilon:
        action = 0
        for i in range(4):
            if Q[state][i] > Q[state][action]: action = i
#         print('greedy', action)
    else:
        action = np.random.choice([0,1,2,3])
#         print('ungreedy', action)
    
    return action


# # Start learning

# In[19]:


Q = np.zeros((16,4))
total_rewards = 0
all_rewards    = []
frames = []
frames_count = 0
episode_reward = 0
episode_count = 0
num_frames = 20000
gamma = 0.8
rate = 0.9
count = 0
state = env.reset()
for frame_idx in range(1, num_frames + 1):
    # get epsilon
    epsilon = epsilon_by_frame(frame_idx)
    
    # forward
    action  = act(state, epsilon) 

    # interact with environment
    env.render()
    next_state, reward, done, info = env.step(action)
    
    # update Q table
    Q[state][action] = Q[state][action] + rate*(reward + gamma*max(Q[next_state]) - Q[state][action])
    
    # go to next state
    state = next_state
    episode_reward += reward
    frames_count += 1
    
    
    if done:
        state = env.reset()
        episode_count += 1
        total_rewards += episode_reward
        all_rewards.append(total_rewards/episode_count)
        frames.append(frames_count)
        frames_count = 0
        episode_reward = 0
        if (reward == 1): count += 1
        print('-----------done')

env.close()
print(episode_count)
print(count)
print(Q)


# In[20]:


plt.plot([all_rewards[i] for i in range(episode_count)])
# print(len(all_rewards))


# In[21]:


plt.plot([frames[i] for i in range(episode_count)])


# In[ ]:




