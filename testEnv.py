
from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
import time
import cv2

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

import gym
import numpy as np
from typing import List


from src.constants import SpiderAndFlyEnv, AgentType, QnetType
from src.agent import Agent
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_qnet_based import QnetBasedAgent
from src.agent_std_rollout import StdRolloutMultiAgent

import warnings

# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)
import wandb
from changeEnv import oddEvenRewardEnv


def create_movie_clip(frames: list, output_file: str, fps: int = 10):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()

frames = []
env = gym.make(SpiderAndFlyEnv)
obs_n = env.reset()
frames.append(env.render())
for i in range(15):
    obs_n, reward_n, done_n, info = env.step([0,0,0,0])
    frames.append(env.render())


create_movie_clip(frames, f"TestMovie_0.mp4", fps=10)

# 0 means move down
# 1 means move left
# 2 means move up
# 3 means move right
# 4 means stay stil
