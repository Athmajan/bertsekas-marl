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
from changeEnv import oddEvenRewardEnv

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
BASIS_AGENT_TYPE = AgentType.RULE_BASED


def convert_to_x(obs, m_agents, agent_id, action_space, prev_actions):
    # state
    obs_first = np.array(obs, dtype=np.float32).flatten()

    # agent ohe
    agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
    agent_ohe[agent_id] = 1.

    # prev actions
    prev_actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)
    for agent_i, action_i in prev_actions.items():
        ohe_action_index = int(agent_i * action_space.n) + action_i
        prev_actions_ohe[ohe_action_index] = 1.

    # combine all
    x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))

    return x



def getSignallingPolicy(net,obs,m_agents,agent_i,prev_actions,action_space):
    agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
    agent_ohe[agent_i] = 1.
    prev_actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)
    if agent_i in prev_actions:
        ohe_action_index = int(agent_i * action_space.n) + prev_actions[agent_i]
        prev_actions_ohe[ohe_action_index] = 1.

    obs_first = np.array(obs, dtype=np.float32).flatten()
    x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))
    xTensor = torch.Tensor(x).view((1,-1))
    best_action = net(xTensor).max(-1)[-1].detach().item()
    return best_action

def getBasePolicy(obs,agent):
    action_id, _ = agent.act_with_info(obs)
    return {agent.id:action_id}





N_SIMS = 10
EPOCHS = 3000

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

    steps_num = 0
    if wandbLog:
        wandb.init(project="Long_Surveillance",name="Autonomous_Offline_MOD")
        log_buffer = []
        log_interval = 50

    _n_workers = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = QNetworkCoordinated(M_AGENTS, P_PREY, 5)
    net.load_state_dict(torch.load(INPUT_QNET_NAME))
    net.to(device)
    net.eval()


    for epi in range(EPOCHS):
        # get episode start time
        startTime = time.time()
        # capture episide frames
        frames = []
        # count steps in an episode
        epi_steps = 0

        # collect total reward of an episode
        total_reward = 0


        obs_n = env.reset()
        frames.append(env.render())

        m_agents = env.n_agents
        p_preys = env.n_preys
        grid_shape = (10,10)
        action_space = env.action_space[0]

        done_n = [False] * m_agents

        while not all(done_n):

            # Query Signalling policy from network eval
            
            prev_actions = {}
            act_n_signalling = []
            
            with ThreadPoolExecutor(max_workers=_n_workers) as executor:
                futures = [
                    executor.submit(getSignallingPolicy, net, obs, m_agents, agent_i, prev_actions, action_space)
                    for agent_i, obs in enumerate(obs_n)
                ]

                for future in as_completed(futures):
                    act_n_signalling.append(future.result())
           
            act_n_signalling_dict = {}
            for i in range(len(act_n_signalling)):
                act_n_signalling_dict[i] = act_n_signalling[i]
            # print(act_n_signalling)

            
            '''
            clearing the role of agents. No need of rule based agents anymore
            we need to work with parallel computations where consequetive agent actions are decided 
            based on rules and the receding agents actions are decided based on the signalling policy.
            This should be just a clone of the role of a sequential rollout agent where the previous actions
            are replaced with the signalling policy than waiting for each agent to communicate.

            Sequential rollout agents are written already to take the future actions to be base policy.
            So only need to give them the previous actions using the signaling policy.
            '''


            agents = [SeqRolloutAgent(
                agent_i, 
                m_agents, 
                p_preys, 
                grid_shape, 
                env.action_space[agent_i],
                n_sim_per_step=N_SIMS, 
                basis_agent_type=BASIS_AGENT_TYPE, 
                qnet_type=QNET_TYPE,
                ) for agent_i in range(m_agents)]
            
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                # here initially doing sequentially.
                # should optimize this further by parallelly
                # comupting actions for all agents at once.
                prev_actions = {}
                for j, (_) in enumerate(zip(agents)):
                    if i > j :
                        prev_actions[j] = act_n_signalling_dict[j]

                action_id = agent.act(obs,modify_env ,prev_actions=prev_actions)

                act_n.append(action_id)

         

            obs_n, reward_n, done_n, info = env.step(act_n)
            epi_steps += 1
            steps_num += 1
            total_reward += np.mean(reward_n)

            frames.append(env.render())
        # end of an episode. capture time    
        endTime = time.time()

        print(f'Episode {epi}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')

        if wandbLog:
            buffered_log({'Reward':total_reward, 'episode_steps' : epi_steps,'exeTime':endTime-startTime}, 
                         epi, log_buffer, log_interval)


        if (epi+1) % 1000 ==0:
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

   



                
                

     




