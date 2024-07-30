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

AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
N_EPISODES = 3000
#GENT_TYPE = AgentType.SEQ_MA_ROLLOUT
QNET_TYPE = QnetType.REPEATED
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 10
SEED = 42
N_SIMS_MC = 50


def create_agents(
        env: gym.Env,
        agent_type: str,
) -> List[Agent]:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    #grid_shape = env.grid_shape
    grid_shape = env.grid_shape

    return [SeqRolloutAgent(
        agent_i, m_agents, p_preys, grid_shape, env.action_space[agent_i],
        n_sim_per_step=N_SIMS, basis_agent_type=BASIS_AGENT_TYPE, qnet_type=QNET_TYPE,
    ) for agent_i in range(m_agents)]
   

# Function to log data with buffering
def buffered_log(data, step, buffer, interval):
    buffer.append((data, step))
    if len(buffer) >= interval:
        for item in buffer:
            wandb.log(item[0], step=item[1], commit=False)
        wandb.log({}, commit=True)  # Commit all logs at once
        buffer.clear()


def main(modify_env,wandbLog):
    if modify_env:
        env = oddEvenRewardEnv(gym.make(SpiderAndFlyEnv),oddRewardScale=10,evenRewardScale=1)
    else:
        env = gym.make(SpiderAndFlyEnv)

    if wandbLog:
        wandb.init(project="Long_Surveillance",name="Sequential Rollout_MOD")
        log_buffer = []
        log_interval = 50
    
    for epi in range (N_EPISODES):
        frames = []
        startTime = time.time()
        obs_n = env.reset()
        agents = create_agents(env, AGENT_TYPE)
        
        done_n = [False] * env.n_agents
        total_reward = 0.
        epi_steps = 0
        while not all(done_n):
            prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                # each agent acts based on the same observation
                action_id = agent.act(obs,modify_env ,prev_actions=prev_actions)
                prev_actions[i] = action_id

                act_n.append(action_id)


            obs_n, reward_n, done_n, info = env.step(act_n)
            epi_steps += 1
            total_reward += np.mean(reward_n)
            frames.append(env.render())

        endTime = time.time()

        print(f'Episode {epi}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')
        if wandbLog:
            buffered_log({'Reward':total_reward, 'episode_steps' : epi_steps,'exeTime':endTime-startTime}, 
                         epi, log_buffer, log_interval)
        


        if (epi+1) % 1000 ==0 and wandbLog:
            print("Checpoint passed")
            # axes are (time, channel, height, width)
            # create_movie_clip(frames, f"ManhattanRuleBased_2_agents_{epi+1}.mp4", fps=10)
            wandb.log({"video": wandb.Video(np.stack(frames,0).transpose(0,3,1,2), fps=10,format="mp4")})

    if wandbLog:
        if log_buffer:
            for item in log_buffer:
                wandb.log(item[0], step=item[1], commit=False)
            wandb.log({}, commit=True)
            log_buffer.clear()
        wandb.finish()
    env.close()


if __name__ == '__main__':
    main(modify_env=True,wandbLog=True)
