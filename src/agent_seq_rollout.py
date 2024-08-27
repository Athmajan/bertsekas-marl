from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gym
import ma_gym

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_rule_based import RuleBasedAgent
from src.agent_qnet_based import QnetBasedAgent, fullQnetBasedAgent
from src.agent import Agent

import matplotlib.pyplot as plt

def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 



class SeqRolloutAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            n_sim_per_step: int = 10,
            basis_agent_type: str = AgentType.RULE_BASED,
            qnet_type: str = None,
            n_workers: int = 12,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space
        self._n_sim_per_step = n_sim_per_step
        self._n_workers = n_workers
        self._agents = self._create_agents(basis_agent_type, qnet_type)

    def act(
            self,
            last_obs_grid,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
            **kwargs,
    ) -> int:
        best_action, action_q_values = self.act_with_info(last_obs_grid,obs, prev_actions)
        return best_action

    def act_with_info1(
            ## with pralallel processing
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
    ) -> Tuple[int, np.ndarray]:
        assert prev_actions is not None

        n_actions = self._action_space.n

        # parallel calculations
        sim_results = []
        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = []

            for action_id in range(n_actions):
                # submit simulation
                futures.append(pool.submit(
                    self._simulate_action_par,

                    self.id,
                    action_id,
                    self._n_sim_per_step,
                    obs,
                    prev_actions,
                    self._m_agents,
                    self._agents,
                ))
            print(futures)
            for f in as_completed(futures):
                print("printing f")
                print(f)
                res = f.result()
                sim_results.append(res)

        # analyze results of the simulation
        np_sim_results = np.array(sim_results, dtype=np.float32)
        np_sim_results_sorted = np_sim_results[np.argsort(np_sim_results[:, 0])]
        action_q_values = np_sim_results_sorted[:, 1]
        best_action = np.argmax(action_q_values)

        return best_action, action_q_values
    
    def act_with_info(
            ## without pralallel processing
            self,
            last_obs_grid,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
        ) -> Tuple[int, np.ndarray]:
        assert prev_actions is not None

        n_actions = self._action_space.n

        # sequential calculations
        sim_results = []

        for action_id in range(n_actions):
            # perform simulation
            res = self._simulate_action_par(
                self.id,
                action_id,
                self._n_sim_per_step,
                obs,
                prev_actions,
                self._m_agents,
                self._agents,
                last_obs_grid,
            )
            sim_results.append(res)

        # analyze results of the simulation
        np_sim_results = np.array(sim_results, dtype=np.float32)
        np_sim_results_sorted = np_sim_results[np.argsort(np_sim_results[:, 0])]
        action_q_values = np_sim_results_sorted[:, 1]
        best_action = np.argmax(action_q_values)

        return best_action, action_q_values

    def _create_agents(
            self,
            agent_type: str,
            qnet_type: str,
    ) -> List[Agent]:
        if agent_type == AgentType.RULE_BASED:
            agents = [RuleBasedAgent(
                i, self._m_agents, self._p_preys, self._grid_shape, self._action_space,
            ) for i in range(self._m_agents)]
        elif agent_type == AgentType.QNET_BASED:
            agents = [QnetBasedAgent(
                i, self._m_agents, self._p_preys, self._grid_shape, self._action_space, qnet_type=qnet_type,
            ) for i in range(self._m_agents)]
        else:
            raise ValueError(f'Invalid agent type: {agent_type}.')

        return agents

    @staticmethod
    def _simulate_action_par(
            agent_id: int,
            action_id: int,
            n_sims: int,
            obs: List[float],
            prev_actions: Dict[int, int],
            m_agents: int,
            agents: List[Agent],
            last_obs_grid,
    ) -> Tuple[int, float]:
        # Memory and CPU load
        # create env
        # run N simulations

        # create env
        env = gym.make(SpiderAndFlyEnv)
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

        # roll first step
        first_step_prev_actions = dict(prev_actions)
        first_act_n = np.empty((m_agents,), dtype=np.int8)
        for i in range(m_agents):
            if i in prev_actions:
                # if ith agent's (other agents) previous action is available
                # use it for env.step
                first_act_n[i] = prev_actions[i]
            elif agent_id == i:
                # if optimization is done for me (not other agents)
                # use my action for env.step
                first_act_n[i] = action_id
                first_step_prev_actions[i] = action_id
            else:
                # if other agents previous actions are not available
                # assume action based on base policy
                underlying_agent = agents[i]
                
                # last_obs_grid[2][env.agent_pos[i][0],env.agent_pos[i][1]] = 1
                actionAss_id, action_distances = underlying_agent.act_with_info_grid(last_obs_grid)
                assumed_action = actionAss_id
                # assumed_action = underlying_agent.act(obs, prev_actions=first_step_prev_actions)
                first_act_n[i] = assumed_action
                first_step_prev_actions[i] = assumed_action

            # last_obs_grid[2] = np.zeros([env.grid_shape[0],env.grid_shape[1]],dtype=np.float32)

        # run N simulations
        avg_total_reward = 0.

        for j in range(n_sims):
            # init env from observation
            env.reset()
            sim_obs_n = env.reset_from(obs)

            # make prescribed first step
            sim_obs_n, sim_reward_n, sim_done_n, sim_info = env.step(first_act_n)
            sim_obs_grid = env.convert_to_flat_obs()
            avg_total_reward += np.sum(sim_reward_n)

            # run simulation
            while not all(sim_done_n):
                sim_act_n = []
                sim_prev_actions = {}
                for agent, sim_obs in zip(agents, sim_obs_n):
                    sim_obs_grid[2][env.agent_pos[agent.id][0],env.agent_pos[agent.id][1]] = 1
                    # sim_best_action = agent.act(sim_obs, prev_actions=sim_prev_actions)
                    sim_best_action, action_distances = agent.act_with_info_grid(sim_obs_grid)
                    sim_act_n.append(sim_best_action)
                    sim_prev_actions[agent.id] = sim_best_action
                    sim_obs_grid = env.convert_to_flat_obs()

                sim_obs_n, sim_reward_n, sim_done_n, sim_info = env.step(sim_act_n)
                avg_total_reward += np.sum(sim_reward_n)

        env.close()

        avg_total_reward /= len(agents)
        avg_total_reward /= n_sims

        return action_id, avg_total_reward





class SeqRolloutAgentQ(Agent):
    def __init__(
            self,
            agent_id: int,
            qnet_name,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
            n_sim_per_step: int = 10,
            n_workers: int = 12,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space
        self._n_sim_per_step = n_sim_per_step
        self._n_workers = n_workers
        # Change the definition of _agents to work with the new model
        self._agents = self._create_agents(qnet_name)

    def act(
            self,
            last_obs_grid,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
            **kwargs,
    ) -> int:
        best_action, action_q_values = self.act_with_info(last_obs_grid,obs, prev_actions)
        return best_action

    def act_with_info1(
            ## with pralallel processing
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
    ) -> Tuple[int, np.ndarray]:
        assert prev_actions is not None

        n_actions = self._action_space.n

        # parallel calculations
        sim_results = []
        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = []

            for action_id in range(n_actions):
                # submit simulation
                futures.append(pool.submit(
                    self._simulate_action_par,

                    self.id,
                    action_id,
                    self._n_sim_per_step,
                    obs,
                    prev_actions,
                    self._m_agents,
                    self._agents,
                ))
            print(futures)
            for f in as_completed(futures):
                print("printing f")
                print(f)
                res = f.result()
                sim_results.append(res)

        # analyze results of the simulation
        np_sim_results = np.array(sim_results, dtype=np.float32)
        np_sim_results_sorted = np_sim_results[np.argsort(np_sim_results[:, 0])]
        action_q_values = np_sim_results_sorted[:, 1]
        best_action = np.argmax(action_q_values)

        return best_action, action_q_values
    
    def act_with_info(
            ## without pralallel processing
            self,
            last_obs_grid,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
        ) -> Tuple[int, np.ndarray]:
        assert prev_actions is not None

        n_actions = self._action_space.n

        # sequential calculations
        sim_results = []
        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = []
            for action_id in range(n_actions):
                futures.append(pool.submit(
                    self._simulate_action_par,
                            self.id,
                            action_id,
                            self._n_sim_per_step,
                            obs,
                            prev_actions,
                            self._m_agents,
                            self._agents,
                            last_obs_grid,
                ))
                for f in as_completed(futures):
                    res = f.result()
                    sim_results.append(res)

        # for action_id in range(n_actions):
        #     # perform simulation
        #     res = self._simulate_action_par(
        #         self.id,
        #         action_id,
        #         self._n_sim_per_step,
        #         obs,
        #         prev_actions,
        #         self._m_agents,
        #         self._agents,
        #         last_obs_grid,
        #     )
        #     sim_results.append(res)

        # analyze results of the simulation
        np_sim_results = np.array(sim_results, dtype=np.float32)
        np_sim_results_sorted = np_sim_results[np.argsort(np_sim_results[:, 0])]
        action_q_values = np_sim_results_sorted[:, 1]
        best_action = np.argmax(action_q_values)

        return best_action, action_q_values

    def _create_agents(
            self,
            qnet_name,
    ) -> List[Agent]:
        agents = [fullQnetBasedAgent(
                i, qnet_name,self._m_agents, self._p_preys, self._grid_shape, self._action_space,
            ) for i in range(self._m_agents)]

        return agents

    @staticmethod
    def _simulate_action_par(
            agent_id: int,
            action_id: int,
            n_sims: int,
            obs: List[float],
            prev_actions: Dict[int, int],
            m_agents: int,
            agents: List[Agent],
            last_obs_grid,
    ) -> Tuple[int, float]:
        # Memory and CPU load
        # create env
        # run N simulations

        # create env
        env = gym.make(SpiderAndFlyEnv)
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

        # roll first step
        first_step_prev_actions = dict(prev_actions)
        first_act_n = np.empty((m_agents,), dtype=np.int8)
        for i in range(m_agents):
            if i in prev_actions:
                # if ith agent's (other agents) previous action is available
                # use it for env.step
                first_act_n[i] = prev_actions[i]
            elif agent_id == i:
                # if optimization is done for me (not other agents)
                # use my action for env.step
                first_act_n[i] = action_id
                first_step_prev_actions[i] = action_id
            else:
                # if other agents previous actions are not available
                # assume action based on base policy
                underlying_agent = agents[i]
                
                # last_obs_grid[2][env.agent_pos[i][0],env.agent_pos[i][1]] = 1
                actionAss_id = underlying_agent.act(last_obs_grid)
                assumed_action = actionAss_id
                # assumed_action = underlying_agent.act(obs, prev_actions=first_step_prev_actions)
                first_act_n[i] = assumed_action
                first_step_prev_actions[i] = assumed_action

            # last_obs_grid[2] = np.zeros([env.grid_shape[0],env.grid_shape[1]],dtype=np.float32)

        # run N simulations
        avg_total_reward = 0.


        for ji in range(n_sims):
            print(ji)
            # init env from observation
            env.reset()
            env.reset_from(obs)
            

            # make prescribed first step
            sim_obs_n, sim_reward_n, sim_done_n, sim_info = env.step(first_act_n)
            sim_obs_grid = env.convert_to_flat_obs()
            avg_total_reward += np.sum(sim_reward_n)
            env.reset_from(sim_obs_n[0])

            try:
                # run simulation
                epiS = 0
                while not all(sim_done_n):
                    sim_act_n = []
                    sim_prev_actions = {}
                    for agent, sim_obs in zip(agents, sim_obs_n):
                        sim_obs_grid[2][env.agent_pos[agent.id][0],env.agent_pos[agent.id][1]] = 1
                        # sim_best_action = agent.act(sim_obs, prev_actions=sim_prev_actions)
                        sim_best_action = agent.act(sim_obs_grid)
                        sim_act_n.append(sim_best_action)
                        sim_prev_actions[agent.id] = sim_best_action
                        sim_obs_grid = env.convert_to_flat_obs()
                    sim_obs_n, sim_reward_n, sim_done_n, sim_info = env.step(sim_act_n)
                    epiS += 1
                    avg_total_reward += np.sum(sim_reward_n)
            except:
                pass

        env.close()

        avg_total_reward /= len(agents)
        avg_total_reward /= n_sims

        return action_id, avg_total_reward