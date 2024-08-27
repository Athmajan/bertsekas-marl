from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
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
from src.agent_qnet_based import QnetBasedAgent, fullQnetBasedAgent, ConvModel


import wandb
import warnings
import cv2
import time
import matplotlib.pyplot as plt
# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)
register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)


SEED = 42
M_AGENTS = 4
P_PREY = 2



N_SAMPLES = 30000
BATCH_SIZE = 1024
EPOCHS = 1000
N_SIMS_MC = 10
FROM_SCRATCH = True
INPUT_QNET_NAME = 'artifacts/basePolicy_50x50_4A_2_RandP.pt'
OUTPUT_QNET_NAME = RepeatedRolloutModelPath_10x10_4v4
BASIS_POLICY_AGENT = AgentType.QNET_BASED
QNET_TYPE = QnetType.BASELINE


def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ConvModel((4,50,50),5)
    net.to(device)
    net.eval()

    net.load_state_dict(torch.load(INPUT_QNET_NAME))

    env = gym.make(SpiderAndFlyEnv)

    done_n = [False] * env.n_agents
    obs_n = env.reset()
    last_obs_grid = env.convert_to_flat_obs()
    frames = []
    epi_steps = 0
    while not all(done_n):
        act_n = []
        for i, obs in enumerate(obs_n):
            qs = net(torch.Tensor(last_obs_grid).unsqueeze(0))
            best_action = qs.max(-1)[-1].item()
            act_n.append(best_action)
            last_obs_grid = env.convert_to_flat_obs()

        print(act_n)
        obs_n, reward_n, done_n, info = env.step(act_n)
        visualize_image(env.render())
        ast_obs_grid = env.convert_to_flat_obs()
        print(epi_steps)
        epi_steps += 1

    print(f"End of {1}'th episode with {epi_steps} steps")
