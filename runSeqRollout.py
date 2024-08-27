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
import matplotlib.pyplot as plt
# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)
import wandb


AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
N_EPISODES = 100
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
   


def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 


if __name__ == '__main__':
    env = gym.make(SpiderAndFlyEnv)


    # env = gym.make(SpiderAndFlyEnv)

    # env.grid_shape=(50, 50),
    # env.n_agents=4,
    # env.n_preys=2,
    # env.prey_move_probs=(0.1, 0.1, 0.1, 0.1, 0.6),
    # #env.prey_move_probs=(13/55, 13/55, 13/55, 13/55,3/55),
    # #env.penalty=1,  # initially -0.5; here we assume no penalty for catching the prey solo
    # env.step_cost=-1,
    # env.prey_capture_reward=0,
    # env.max_steps=5000,
    # env.smartPrey = False
    # env._grid_shape = (50,50)


    wandb.init(project="smartFlies",name="Seq_50x50_4A_10_A*_P")
    
    for epi in range (N_EPISODES):
        frames = []
        startTime = time.time()
        obs_n = env.reset()
        last_obs_grid = env.convert_to_flat_obs()
        agents = create_agents(env, AGENT_TYPE)
        
        done_n = [False] * env.n_agents
        total_reward = 0.
        epi_steps = 0
        while not all(done_n):
            prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                last_obs_grid[2][env.agent_pos[agent.id][0],env.agent_pos[agent.id][1]] = 1
                # each agent acts based on the same observation
                action_id = agent.act(last_obs_grid,obs, prev_actions=prev_actions)
                prev_actions[i] = action_id

                act_n.append(action_id)

                last_obs_grid = env.convert_to_flat_obs()


            obs_n, reward_n, done_n, info = env.step(act_n)
            print(act_n)
            print(info)
            epi_steps += 1
            total_reward += np.mean(reward_n)
            frames.append(env.render())
            # visualize_image(env.render())

        endTime = time.time()

        print(f'Episode {epi}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')
        wandb.log({'Reward':total_reward, 'episode_steps' : epi_steps,'exeTime':endTime-startTime},step=epi) 
        


        if (epi+1) % 10 ==0:
            print("Checpoint passed")
            # axes are (time, channel, height, width)
            # create_movie_clip(frames, f"ManhattanRuleBased_2_agents_{epi+1}.mp4", fps=10)
            wandb.log({"video": wandb.Video(np.stack(frames,0).transpose(0,3,1,2), fps=20,format="mp4")})

    wandb.finish()
    env.close()