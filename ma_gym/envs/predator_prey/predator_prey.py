import copy
import logging
from typing import List, Tuple

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.observation_space import MultiAgentObservationSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text

logger = logging.getLogger(__name__)
import heapq

GRID_SIZE = 50
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1), (0, 0)]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos):
    neighbors = []
    for action in ACTIONS:
        next_pos = (pos[0] + action[0], pos[1] + action[1])
        if 0 <= next_pos[0] < GRID_SIZE and 0 <= next_pos[1] < GRID_SIZE:
            neighbors.append(next_pos)
    return neighbors

def a_star_search(start, predators):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: min(heuristic(start, pred) for pred in predators)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current in predators:
            continue  # Skip if the position is a predator
        
        neighbors = get_neighbors(current)
        
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1  # Assume cost of moving is 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + min(heuristic(neighbor, pred) for pred in predators)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
    return reconstruct_path(came_from, start)

def reconstruct_path(came_from, start):
    current = start
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def find_safest_move(prey_pos, predators):
    path = a_star_search(prey_pos, predators)
    if len(path) > 1:
        next_move = path[1]
        dx = next_move[0] - prey_pos[0]
        dy = next_move[1] - prey_pos[1]
        action = ACTIONS.index((dx, dy))
    else:
        action = 4  # No-op if no safe path found
    return action


def scan_grid(next_pos, grid_size):
        positions_to_scan = []
        for dx in range(-grid_size, grid_size + 1):
            for dy in range(-grid_size, grid_size + 1):
                if dx != 0 or dy != 0:  # Exclude the center position
                    positions_to_scan.append([next_pos[0] + dx, next_pos[1] + dy])
        return positions_to_scan

class PredatorPrey(gym.Env):
    """
    AAA
    Here there are m spiders and one fly moving on a
    2-dimensional grid. During each time period the fly moves
    to some other position according to a given state-dependent
    probability distribution. The spiders, working as a team, aim
    to catch the fly at minimum cost (thus the one-stage cost is
    equal to 1, until reaching the state where the fly is caught,
    at which time the one-stage cost becomes 0). Each spider
    learns the current state (the vector of spiders and fly locations)
    at the beginning of each time period, and either moves to a
    neighboring location or stays where it is. [Bertsekas, 2020]
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            grid_shape=(50, 50), 
            n_agents=4, 
            n_preys=2,
            prey_move_probs=(0.1, 0.1, 0.1, 0.1, 0.6), 
            step_cost=-1,
            prey_capture_reward=0, 
            max_steps=5000, 
            smartPrey=False):
        

        self._grid_shape = grid_shape
        self.grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.max_steps = max_steps
        self._step_count = None
        self._substep_count = 0
        # self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}
        self._prey_alive = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()

        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
        self.prey_move_probs = prey_move_probs
        self.viewer = None

        # Returns relative position -> positions of all agents & prey
        self._obs_high = np.array([1., 1.] * n_agents + [1., 1.] * n_preys)
        self._obs_low = np.array([0., 0.] * n_agents + [0., 0.] * n_preys)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

        self.smartPrey = smartPrey

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_positions(self):
        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos, agent_id=agent_i):
                    self.agent_pos[agent_i] = pos
                    break

        for prey_i in range(self.n_preys):
            while True:
                pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                       self.np_random.randint(0, self._grid_shape[1] - 1)]

                if self._is_cell_vacant(pos, prey_id=prey_i):  # and (self._neighbour_agents(pos)[0] == 0):
                    self.prey_pos[prey_i] = pos
                    break

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        
        # all agents' position
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            #print(pos)

            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates
            #print("Printing Agent's Observations")
            #print(_agent_i_obs)
            _obs.append(_agent_i_obs)

        # all preys' position
        for prey_j in range(self.n_preys):
            pos = self.prey_pos[prey_j]
            _prey_j_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            _obs.append(_prey_j_obs)

        # alive status fro prey
        _obs.append([1. if alive else 0. for alive in self._prey_alive])

        #print(_obs)
        # same observations for all agents
        #_obs = np.array(_obs).flatten().tolist() # this does not work for n_preys != 2 since other items are of length 2
        _obs = np.array([item for sublist in _obs for item in sublist])
        _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        self.__init_positions()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]

        return self.get_agent_obs()

    def reset_default(self):
        print("Number of preys",self.n_preys)
        print("Number of agents",self.n_agents)

        assert self.n_preys == 2
        assert self.n_agents == 4
        assert self._grid_shape == (10, 10)

        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        # closer to the second prey
        self.agent_pos[0] = [9, 5]
        self.agent_pos[1] = [9, 6]
        self.agent_pos[2] = [5, 9]
        self.agent_pos[3] = [6, 9]

        # at corners
        self.prey_pos[0] = [0, 0]
        self.prey_pos[1] = [9, 9]

        self.__draw_base_img()

        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]

        return self.get_agent_obs()

    def reset_from(self, obs):
        # assert self.n_preys == 2
        # assert self.n_agents == 4
        # assert self._grid_shape == (10, 10)

        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        # agent positions
        for i in range(self.n_agents):
            start_ind = int(i * 2)
            row_pos_scaled, col_pos_scaled = obs[start_ind], obs[start_ind + 1]
            row_pos, col_pos = self._convert_to_pos((row_pos_scaled, col_pos_scaled))
            self.agent_pos[i] = [row_pos, col_pos]

        # prey positions
        for j in range(self.n_preys):
            start_ind = int(self.n_agents * 2 + j * 2)
            row_pos_scaled, col_pos_scaled = obs[start_ind], obs[start_ind + 1]
            row_pos, col_pos = self._convert_to_pos((row_pos_scaled, col_pos_scaled))
            self.prey_pos[j] = [row_pos, col_pos]

        self.__draw_base_img()

        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]

        preys_alive = obs[-self.n_preys:]
        self._prey_alive = [bool(prey_alive) for prey_alive in preys_alive]

        return self.get_agent_obs()
    
    def convert_to_flat_obs(self):
        prey_alive_status = self._prey_alive
        agent_pos = self.agent_pos
        prey_pos = self.prey_pos
        gridSize = self.grid_shape


        # create empty array with the grid size
        grid_agent_dones = np.zeros([gridSize[0],gridSize[1]],dtype=np.float32)
        grid_agent_pos = np.zeros([gridSize[0],gridSize[1]],dtype=np.float32)
        grid_agent_pos_wID = np.zeros([gridSize[0],gridSize[1]],dtype=np.float32)

        # create grid array with agent positions
        # mark 1 if any agent is present 
        for agentID in range(self.n_agents):
            agentLoc = agent_pos[agentID]
            agentLoc_x = agentLoc[0]
            agentLoc_y = agentLoc[1]

            grid_agent_pos[agentLoc_x,agentLoc_y] = 1
            
            # grid_agent_pos_wID[agentLoc_x,agentLoc_y] = agentID+1 # adding one to distinguish between default 0


            agentDone = self._agent_dones[agentID]
            if agentDone:
                grid_agent_dones[agentLoc_x,agentLoc_y] = 1

        # create empty array with the grid size
        grid_prey_alive = np.zeros([gridSize[0],gridSize[1]],dtype=np.float32)

        for preyID in range(self.n_preys):
            preyLoc = prey_pos[preyID]
            preyLoc_x = preyLoc[0]
            preyLoc_y = preyLoc[1]

            preyAlive = prey_alive_status[preyID]
            if preyAlive:
                grid_prey_alive[preyLoc_x,preyLoc_y] = 1


        grid_stack = np.stack((grid_agent_dones, grid_agent_pos, grid_agent_pos_wID,grid_prey_alive), axis=0)


        return grid_stack

    def _convert_to_pos(
            self,
            pos_scaled: Tuple[float, float],
    ) -> Tuple[int, int]:
        grid_row, grid_col = self._grid_shape
        row_pos_scaled, col_pos_scaled = pos_scaled
        row_pos = int(np.round((grid_row - 1) * row_pos_scaled, 0))
        col_pos = int(np.round((grid_col - 1) * col_pos_scaled, 0))
        return row_pos, col_pos

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos, agent_id=None, prey_id=None):
        assert (agent_id is not None) or (prey_id is not None)

        if not self.is_valid(pos):
            return False

        # check that position does not intersect with the existing agents
        for i, pos_i in self.agent_pos.items():
            if (agent_id is not None) and agent_id == i:
                continue

            if pos_i[0] == pos[0] and pos_i[1] == pos[1]:
                return False

        # check that position does not intersect with the existing preys
        for j, pos_j in self.prey_pos.items():
            if (prey_id is not None) and prey_id == j:
                continue

            if pos_j[0] == pos[0] and pos_j[1] == pos[1]:
                return False

        return True

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self.is_valid(next_pos):  # self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0], curr_pos[1]-1]
        elif move == 1:  # left
            next_pos = [curr_pos[0]-1, curr_pos[1]]
        elif move == 2:  # up
            next_pos = [curr_pos[0], curr_pos[1]+1]
        elif move == 3:  # right
            next_pos = [curr_pos[0]+1, curr_pos[1]]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self.is_valid(next_pos):  # self._is_cell_vacant(next_pos):
                self.prey_pos[prey_i] = next_pos
            else:
                # print('pos not updated')
                pass

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id
    


    
    

    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."
        
        self._step_count += 1

        rewards = [self._step_cost for _ in range(self.n_agents)]

        # all agents move
        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        '''
        Making a modification to the envionment here from the original (main)
        where the preys become smart. They simulate their options for the next move.
        The simualted next step will consider how many agents are there in the neighbourhood.
        And it will take the most safest move.
        '''

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:

                if self.smartPrey :
                    predator_positions = []
                    for value in self.agent_pos.values():
                        predator_positions.append(value)


                    prey_move = find_safest_move(tuple(self.prey_pos[prey_i]), predator_positions)

                else:
                    prey_move = self.np_random.choice(len(self._prey_move_probs),1, p=self._prey_move_probs)[0]

                self.__update_prey_pos(prey_i,prey_move )

                # recalculate alive status + add reward if caught
                prey_j_pos = self.prey_pos[prey_i]
                for agent_i, agent_pos in self.agent_pos.items():
                    # do not add several rewards if caught by multiple agents
                    if self._prey_alive[prey_i]:
                        if prey_j_pos[0] == agent_pos[0] and prey_j_pos[1] == agent_pos[1]:
                            self._prey_alive[prey_i] = False
                            rewards[agent_i] += self._prey_capture_reward



        if (self._step_count >= self.max_steps) or (True not in self._prey_alive):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

    def substep(self, agent_id, action):
        self._substep_count += 1

        # one agent moves
        if not (self._agent_dones[agent_id]):
            self.__update_agent_pos(agent_id, action)

        is_full_step = self._substep_count % self.n_agents == 0

        # Partial Step -> update only one agent's position
        if not is_full_step:
            return self.get_agent_obs(), None, None, {'prey_alive': self._prey_alive}
        # Full Step -> update prey, recalculate rewards
        else:
            self._step_count += 1

            rewards = [self._step_cost for _ in range(self.n_agents)]

            # all preys move
            for prey_i in range(self.n_preys):
                if self._prey_alive[prey_i]:
                    _move = self.np_random.choice(len(self._prey_move_probs), 1, p=self._prey_move_probs)[0]
                    self.__update_prey_pos(prey_i, _move)

                    # recalculate alive status + add reward if caught
                    prey_j_pos = self.prey_pos[prey_i]
                    for agent_i, agent_pos in self.agent_pos.items():
                        # do not add several rewards if caught by multiple agents
                        if self._prey_alive[prey_i]:
                            if prey_j_pos[0] == agent_pos[0] and prey_j_pos[1] == agent_pos[1]:
                                self._prey_alive[prey_i] = False
                                rewards[agent_i] += self._prey_capture_reward

            if (self._step_count >= self.max_steps) or (True not in self._prey_alive):
                for i in range(self.n_agents):
                    self._agent_dones[i] = True

            for i in range(self.n_agents):
                self._total_episode_reward[i] += rewards[i]

            return self.get_agent_obs(), rewards, self._agent_dones, {'prey_alive': self._prey_alive}

    def apply_move(self, agent_id, action):
        # one agent moves
        if not (self._agent_dones[agent_id]):
            self.__update_agent_pos(agent_id, action)

        return self.get_agent_obs()

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='rgb_array'):
        img = copy.copy(self._base_img)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(img, text=str(prey_i + 1), pos=self.prey_pos[prey_i], cell_size=CELL_SIZE,
                                fill='white', margin=0.4)

        # for agent_i in range(self.n_agents):
        #     for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
        #         fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
        #     fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        img = np.asarray(img)
        #print(img)
        return img
        # if mode == 'rgb_array':
        #     return img
        # elif mode == 'human':
        #     from gym.envs.classic_control import rendering
        #     if self.viewer is None:
        #         self.viewer = rendering.SimpleImageViewer()
        #     self.viewer.imshow(img)
        #     return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def apply_action(self, curr_pos, move):
        # curr_pos = copy.copy(self.agent_pos[agent_i])
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = [curr_pos[0], curr_pos[1]]
        else:
            raise Exception('Action Not found!')

        return next_pos if self.is_valid(next_pos) else None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}
