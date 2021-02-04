from time import perf_counter

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from agents.constants import SpiderAndFlyEnv, BaselineModelPath_10x10_4v2
from agents.baseline_agent import BaselineAgent
from agents.qnetwork_std_rollout import QNetworkStdRollout

N_AGENTS = 4
N_PREY = 2
ACTION_SPACE = 5

N_SAMPLES = 100_000
BATCH_SIZE = 128
EPOCHS = 10

SEED = 42


def generate_samples(n_samples, seed):
    print('Started sample generation.')

    samples = []

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    env.seed(seed)

    while len(samples) < n_samples:
        # init env
        obs_n = env.reset()

        # init agents
        n_agents = env.n_agents
        n_preys = env.n_preys
        agents = [BaselineAgent(i, n_agents, n_preys, env.action_space[i]) for i in range(n_agents)]

        # init stopping condition
        done_n = [False] * n_agents

        # run 100 episodes for a random agent
        while not all(done_n):
            # for each agent calculates Manhattan Distance to each prey for each
            # possible action
            # O(n*m*q)
            distances = env.get_distances()

            # transform into samples
            obs_first = np.array(obs_n[0]).flatten()  # same for all agent

            agent_min_distances = []

            for i, a_dist in enumerate(distances):
                min_prey = a_dist.min(axis=1)
                agent_min_distances.append(min_prey)

            samples.append((obs_first, -np.concatenate(agent_min_distances)))

            # all agents act based on the observation
            act_n = []
            for agent, obs, action_distances in zip(agents, obs_n, distances):
                max_action = agent.act(action_distances)
                act_n.append(max_action)

            # update step ->
            obs_n, reward_n, done_n, info = env.step(act_n)

        env.close()

    print('Finished sample generation.')

    return samples[:N_SAMPLES]


def train_qnetwork(samples):
    print('Started Training.')

    net = QNetworkStdRollout(N_AGENTS, N_PREY, ACTION_SPACE)

    net.train()  # check

    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        n_batches = 0

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())

            loss = criterion(outputs.float(), labels.float())

            loss.backward()

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            n_batches += 1

        # if epoch % 10 == 0:
        #     torch.save(net.state_dict(), BaselineModelPath_10x10_4v2)

        print(f'[{epoch}] {running_loss / n_batches:.3f}.')

    print('Finished Training.')

    return net


if __name__ == '__main__':
    # fix seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # run experiment
    t1 = perf_counter()
    train_samples = generate_samples(N_SAMPLES, SEED)

    t2 = perf_counter()
    net = train_qnetwork(train_samples)
    t3 = perf_counter()

    # save
    torch.save(net.state_dict(), BaselineModelPath_10x10_4v2)

    print(f'Generated samples in {(t2 - t1) / 60.:.2f} min.')
    print(f'Trained in {(t3 - t2) / 60.:.2f} min.')
