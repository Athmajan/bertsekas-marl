'''
This is an attempt to learn the base policy using the rulebased policy experiences.
The observations given by the the original enviorment is unable to generalize for more flies and spiders.
So trying to flatten the complete gridworld with the spiders' and flies' positions.
'''
from gym.envs.registration import register
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from random import sample, random
import numpy as np
import ipdb
from src.constants import SpiderAndFlyEnv
import gym

from src.agent_rule_based import RuleBasedAgent
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)


class ConvModel(nn.Module):
    def __init__(self,obs_shape,num_actions,lr=0.0001):
        assert len(obs_shape) ==3 # channel, height and width
        super(ConvModel,self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        # Canonical input size is 84x84
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(4,16,(8,8),stride=(4,4)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16,32,(4,4),stride=(2,2)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape)) # 1 is the batch size
            x = self.conv_net(dummy)
            s = x.shape
            fc_size = s[1] * s[2] * s[3]


        self.fc_net  = torch.nn.Sequential(
            torch.nn.Linear(fc_size,256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),

            torch.nn.Linear(256,64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),


            torch.nn.Linear(64,num_actions),
        )


    def forward(self,x):
        # import ipdb; ipdb.set_trace()
        conv_latent = self.conv_net(x) # shape : (N, ___)
        # we need to keep the batch dimension the same 
        # and flatten the rest
        return self.fc_net(conv_latent.view((conv_latent.shape[0],-1)))
    

class ReplayBuffer:
    def __init__(self,buffer_size = 100000):
        self.buffer_size = buffer_size
        self.buffer = [None]*buffer_size # fixed size array
        self.idx = 0

    def insert(self,sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        
        assert num_samples < min(self.idx,self.buffer_size)

        if self.idx < self.buffer_size:
        # until we reach the buffer size we cant  sample
        # from the entire array but sample upto idx only
            return sample(self.buffer[:self.idx],num_samples)
        return sample(self.buffer,num_samples)
            
        # return sample(self.buffer, num_samples)


@dataclass
class Sarsd:
    state: Any
    action : int
    reward : float
    next_state : Any
    done : bool

def update_tgt_model(m,tgt):
    '''
    THis is to copy the weights from one to another
    '''
    tgt.load_state_dict(m.state_dict())


def OHE_AgentID(env,last_obs_grid,agent):
    last_obs_grid[2][env.agent_pos[agent.id][0],env.agent_pos[agent.id][1]] = 1
    return last_obs_grid



EPOCHS = 100
BATCH_SIZE = 1000
OUTPUT_QNET_NAME =  'artifacts/basePolicy_50x50_4A_2_RandP.pt'
N_SAMPLES = 10000

def generate_sample(n_samples):
    env = gym.make(SpiderAndFlyEnv)
    samples = []
    while len(samples) < n_samples:
        obs_n = env.reset()
        last_obs_grid = env.convert_to_flat_obs()
        print(len(samples))
        agents = [RuleBasedAgent(i, env.n_agents, env.n_preys, env.grid_shape, env.action_space[0]) for i in range(env.n_agents)]
        done_n = [False] * env.n_agents
        while not all(done_n):
            act_n = []
            for i, agent in enumerate(agents):
                #OHE 
                last_obs_grid[2][env.agent_pos[agent.id][0],env.agent_pos[agent.id][1]] = 1
                
                action_id, action_q_values = agent.act_with_info_grid(last_obs_grid)
                min_val = np.min(action_q_values)
                max_val = np.max(action_q_values)
                try:
                    min_max_normalized_q_values = (action_q_values - min_val) / (max_val - min_val)
                except:
                    min_max_normalized_q_values = action_q_values/max_val


                act_n.append(action_id)
                # resettting OHE
                last_obs_grid = env.convert_to_flat_obs()
                samples.append((last_obs_grid, min_max_normalized_q_values))

            obs_n, reward_n, done_n, info = env.step(act_n)
            new_obs_grid = env.convert_to_flat_obs()
            last_obs_grid = new_obs_grid

    return samples[:n_samples]

def train_qnetwork(samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ConvModel((4,50,50),5)
    net.to(device)
    net.train()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    
    for epoch in range(EPOCHS):
        running_loss = .0
        n_batches = 0
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(running_loss)
            n_batches += 1

        if epoch % 10 == 0:
            torch.save(net.state_dict(), OUTPUT_QNET_NAME)

    return net



if __name__ == '__main__':
    # m = ConvModel((4,50,50),5)
    # tensor = torch.zeros((1,4,50,50))
    # print(m(tensor))
    # print(m.forward(tensor))

    # print(m.forward(tensor).max(-1)[-1].item())

    # env = gym.make(SpiderAndFlyEnv)
    
    # env.n_agents = 4
    # env.n_preys = 10
    # env.max_steps = 5000
    # env._grid_shape = (50,50)
    # env.grid_shape = (50,50)
    # env.smartPrey = False


    # m = ConvModel((4,50,50),5)
    # tgt = ConvModel((4,50,50),5)
    # update_tgt_model(m,tgt)

    # rb = ReplayBuffer()
    # epsilon = 1.0  # Initial exploration rate
    # epsilon_min = 0.01  # Minimum exploration rate
    # epsilon_decay = 0.995  # Decay rate for epsilon

    # for epi in range(EPOCHS):
    #     env.reset()
    #     last_obs_grid = env.convert_to_flat_obs()


    #     agents = [RuleBasedAgent(i, env.n_agents, env.n_preys, env.grid_shape, env.action_space[0]) for i in range(env.n_agents)]
        
    #     done_n = [False] * env.n_agents

    #     while not all(done_n):

    #         act_n = []
    #         for agent in agents:
    #             # OHE of agent
    #             last_obs_grid[2][env.agent_pos[agent.id][0],env.agent_pos[agent.id][1]] = 1
    #             if np.random.rand() <= epsilon:
    #                 action_id, action_distances = agent.act_with_info_grid(last_obs_grid)
    #                 last_obs_grid = env.convert_to_flat_obs()
    #             else:
    #                 action_id = m.forward(torch.Tensor(last_obs_grid).unsqueeze(0)).max(-1)[-1].item()

    #             act_n.append(action_id)

    #             # reset the OHE layer before moving to the other agent
    #             last_obs_grid = env.convert_to_flat_obs()


    #         obs_n, reward_n, done_n, info = env.step(act_n)
    #         new_obs_grid = env.convert_to_flat_obs()


    #         if epsilon > epsilon_min:
    #             epsilon *= epsilon_decay

    n_workers = 8
    chunk = int(N_SAMPLES / n_workers)
    train_samples = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:

        futures = []
        for _ in range(n_workers):
            futures.append(pool.submit(generate_sample, chunk))

        for f in as_completed(futures):
            samples_part = f.result()
            train_samples += samples_part

    # train_samples = generate_sample(N_SAMPLES)
    net = train_qnetwork(train_samples)
    torch.save(net.state_dict(), OUTPUT_QNET_NAME)





                