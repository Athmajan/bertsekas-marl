from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, RepeatedRolloutModelPath_10x10_4v4, AgentType, \
    QnetType
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_seq_rollout import SeqRolloutAgent, RuleBasedAgent
from src.agent_qnet_based import QnetBasedAgent


import wandb
import warnings
import cv2


# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)


SEED = 42
M_AGENTS = 4
P_PREY = 2



N_SAMPLES = 30000
BATCH_SIZE = 1024
EPOCHS = 1000
N_SIMS_MC = 10
FROM_SCRATCH = True
INPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v4
OUTPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v4
BASIS_POLICY_AGENT = AgentType.QNET_BASED
QNET_TYPE = QnetType.BASELINE


def create_movie_clip(frames: list, output_file: str, fps: int = 10):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()


if __name__ == '__main__':
    env = gym.make(SpiderAndFlyEnv)
    env._grid_shape = (50,50)
    env.max_steps = 5000
    
    
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env._grid_shape
    action_space = env.action_space[0]

    agents = [QnetBasedAgent(i, m_agents, p_preys, grid_shape, env.action_space[i]) for i in range(m_agents)]

    done_n = [False] * m_agents
    obs_n = env.reset()
    frames = []
    while not all(done_n):
        prev_actions = {}
        act_n = []
        for i, (agent, obs) in enumerate(zip(agents, obs_n)):
            best_action = agent.act(obs,prev_actions)

            prev_actions[i] = best_action
            act_n.append(best_action)

        obs_n, reward_n, done_n, info = env.step(act_n)
        frames.append(env.render())

    env.close()
    create_movie_clip(frames,'QnetBasepolicy.mp4')

    
