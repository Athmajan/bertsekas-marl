from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, BaselineModelPath_10x10_4v2
from src.qnetwork import QNetwork
from src.agent_rule_based import RuleBasedAgent

import cv2
import wandb
SEED = 42
M_AGENTS = 4
P_PREY = 2

EPOCHS = 3000
import time

from changeEnv import oddEvenRewardEnv

def create_movie_clip(frames: list, output_file: str, fps: int = 10):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()


def main(modify_env,wandbLog):
    if modify_env:
        env = oddEvenRewardEnv(gym.make(SpiderAndFlyEnv),oddRewardScale=10,evenRewardScale=1)
    else:
        env = gym.make(SpiderAndFlyEnv)

    steps_history = []
    
    steps_num = 0

    if wandbLog:
        wandb.init(project="Long_Surveillance",name="ManhattanDistanceRule_MOD")

    for epi in range(EPOCHS):
        startTime = time.time()
        frames = []
        epi_steps = 0
        obs_n = env.reset()
        frames.append(env.render())
        m_agents = env.n_agents
        p_preys = env.n_preys
        grid_shape = (10, 10)
        
        agents = [RuleBasedAgent(i, m_agents, p_preys, grid_shape, env.action_space[i]) for i in range(m_agents)]

        done_n = [False] * m_agents
        total_reward = 0
        while not all(done_n):
            obs_first = np.array(obs_n[0], dtype=np.float32).flatten()
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                # each agent is passed the same observation and asked to act
                agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
                agent_ohe[i] = 1.
                action_id, action_distances = agent.act_with_info(obs)
                
                
                epi_steps += 1
                steps_num += 1

                act_n.append(action_id)

            obs_n, reward_n, done_n, info = env.step(act_n)
            total_reward += np.mean(reward_n)
            frames.append(env.render())

        endTime = time.time()
        if wandbLog:
            wandb.log({'Reward':total_reward, 'episode_steps' : epi_steps,'exeTime':endTime-startTime},step=epi) 
        print(f"End of {epi}'th episode with {epi_steps} steps")
        steps_history.append(epi_steps)
        

        if (epi+1) % 1000 ==0 and wandbLog:
            print("Checpoint passed")
            # axes are (time, channel, height, width)
            # create_movie_clip(frames, f"ManhattanRuleBased_2_agents_{epi+1}.mp4", fps=10)
            wandb.log({"video": wandb.Video(np.stack(frames,0).transpose(0,3,1,2), fps=10,format="mp4")})
            


    if wandbLog:
        wandb.finish()
    env.close()

    # create_movie_clip(frames, 'ManhattanRuleBased_2_agents.mp4', fps=10)

            



if __name__ == "__main__":
    main(modify_env = False, wandbLog = False)
    
        