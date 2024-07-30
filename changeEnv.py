'''
This is an attempt to write a wrapper to the existing environment
to create changed in the enviornment to test the robustness/ resilliency 
of the developed solutions
'''

import time
from typing import List, Iterable

from tqdm import tqdm
import numpy as np
from numpy import random
import gym
import ma_gym  # register new envs on import
import torch
import torch.nn as nn
import torch.optim as optim

from src.constants import SpiderAndFlyEnv, RolloutModelPath_10x10_4v2
from src.qnetwork import QNetwork
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_rule_based import RuleBasedAgent
from src.agent_qnet_based import QnetBasedAgent

N_EPISODES = 20
N_SIMS_PER_STEP = 10
BATCH_SIZE = 512
EPOCHS = 10

import warnings
import logging
import cv2
import matplotlib.pyplot as plt


from gym.envs.registration import register

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

gym.logger.set_level(logging.ERROR)  # Set gym logger level to ERROR

warnings.filterwarnings("ignore", category=UserWarning, module="gym")  # Ignore UserWarnings from gym


class oddEvenRewardEnv():
    '''
    Example : This wrapper will reward -10 for agents ids which are odd
    and -1 for agent ids which are even
    '''
    def __init__(self,env,oddRewardScale,evenRewardScale):
        self.env = env
        self.evenRew = evenRewardScale
        self.oddRew = oddRewardScale

        
    def step(self,act_n):
        obs_n, reward_n, done_n, info = self.env.step(act_n)
        new_reward_n = []
        for i,rew in enumerate(reward_n):
            if i % 2 == 0:
                # even
                new_reward_n.append(rew*self.evenRew)
            else:
                # odd
                new_reward_n.append(rew*self.oddRew)

        return obs_n, new_reward_n, done_n, info

    def __getattr__(self, name):
        return getattr(self.env, name)
    
    def __setattr__(self, name, value):
        if name in {'env', 'evenRew', 'oddRew'}:
            super().__setattr__(name, value)
        else:
            setattr(self.env, name, value)
    





    


if __name__ == '__main__':
    env = gym.make(SpiderAndFlyEnv)
    env = oddEvenRewardEnv(env,oddRewardScale=10,evenRewardScale=1)


    env.seed(42)

    m_agents = env.n_agents
    p_preys = env.n_preys
    #grid_shape = env._grid_shape
    action_space = env.action_space[0]
    done_n = [False] * env.n_agents
    env.reset()
    total_reward = 0
    epi_steps = 0
    while not all(done_n):
        obs_n, reward_n, done_n, info = env.step(random.choice([0,1,2,3,4], size = 4))
        print(reward_n)
        total_reward += np.sum(reward_n)
        epi_steps += 1

    env.close()

    print(f"Episode with reward {total_reward/m_agents} and steps {epi_steps}")    

