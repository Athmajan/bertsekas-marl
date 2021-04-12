import time
from typing import List

import gym
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_exact_rollout import ExactRolloutAgent
from src.agent_approx_rollout import RolloutAgent

N_EPISODES = 3
AGENT_TYPE = AgentType.APRX_ROLLOUT


def create_agents(env: gym.Env, agent_type: str) -> List:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    grid_shape = env._grid_shape

    if agent_type == AgentType.RANDOM:
        agents = [RandomAgent(env.action_space[i]) for i in range(m_agents)]
    elif agent_type == AgentType.RULE_BASED:
        agents = [RuleBasedAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    elif agent_type == AgentType.EXACT_ROLLOUT:
        agents = [ExactRolloutAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    elif agent_type == AgentType.APRX_ROLLOUT:
        agents = [RolloutAgent(i, m_agents, p_preys, grid_shape, env.action_space[i])
                  for i in range(m_agents)]
    else:
        raise ValueError(f'Unrecognized agent type: {agent_type}.')

    return agents


if __name__ == '__main__':
    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)
    # env = Monitor(env, directory='../artifacts/recordings', force=True,)

    for _ in range(N_EPISODES):
        # init env
        # obs_n = env.reset()
        obs_n = env.reset_default()
        env.render()

        # init agents
        agents = create_agents(env, AGENT_TYPE)

        # init stopping condition
        done_n = [False] * env.n_agents

        # run an episode until all prey is caught
        while not all(done_n):
            prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                action_id = agent.act(obs, prev_actions=prev_actions)

                prev_actions[i] = action_id
                act_n.append(action_id)

            # update step
            obs_n, reward_n, done_n, info = env.step(act_n)

            time.sleep(0.5)
            env.render()

    time.sleep(2.)

    env.close()