from time import perf_counter
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple

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
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_rule_based import RuleBasedAgent

import time
import wandb
from src.agent import Agent


import warnings

# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)


SEED = 42

M_AGENTS = 4
P_PREY = 2

N_SAMPLES = 50_000
BATCH_SIZE = 1024
EPOCHS = 500
N_SIMS_MC = 50
FROM_SCRATCH = False
INPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v4
BASIS_POLICY_AGENT = AgentType.QNET_BASED
QNET_TYPE = QnetType.BASELINE




N_SIMS = 10
EPOCHS = 30




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = QNetworkCoordinated(M_AGENTS, P_PREY, 5)
    net.load_state_dict(torch.load(INPUT_QNET_NAME))
    net.to(device)
    net.eval()

    env = gym.make(SpiderAndFlyEnv)
    obs_n = env.reset()

    


