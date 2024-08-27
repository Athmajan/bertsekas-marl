from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
import time
import cv2
import ipdb

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
from src.agent_seq_rollout import SeqRolloutAgentQ
from src.agent_qnet_based import QnetBasedAgent
from src.agent_std_rollout import StdRolloutMultiAgent

import warnings
import matplotlib.pyplot as plt
# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)
import wandb

N_EPISODES = 10
N_SIMS = 5
IN_QNET = 'artifacts/basePolicy_50x50_4A_2_RandP.pt'
N_WORK = 8


if __name__ == "__main__":
    
    env = gym.make(SpiderAndFlyEnv)
    for epi in range (N_EPISODES):
        frames = []
        startTime = time.time()
        obs_n = env.reset()
        last_obs_grid = env.convert_to_flat_obs()
        done_n = [False] * env.n_agents

        agents =[SeqRolloutAgentQ(
                agent_i, IN_QNET,env.n_agents, env.n_preys, env.grid_shape, env.action_space[0],
                n_sim_per_step=N_SIMS,n_workers=N_WORK
                ) for agent_i in range(env.n_agents)]
        
        total_reward = 0.
        epi_steps = 0
        while not all(done_n):
            prev_actions = {}
            act_n = []
            for i, agent in enumerate(agents):
                # OHE
                last_obs_grid[2][env.agent_pos[agent.id][0],env.agent_pos[agent.id][1]] = 1

                action_id = agent.act(last_obs_grid,obs_n[0], prev_actions=prev_actions)


                prev_actions[i] = action_id
                last_obs_grid = env.convert_to_flat_obs()
                act_n.append(action_id)

            obs_n, reward_n, done_n, info = env.step(act_n)
            epi_steps += 1
            print("epi_steps",epi_steps)
            total_reward += np.mean(reward_n)
            frames.append(env.render())

        endTime = time.time()
        print(f'Episode {epi}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')

    env.close()




