from typing import Tuple, List, Dict

import numpy as np
import gym

from src.agent import Agent
from src.qnetwork_coordinated import QNetworkCoordinated
from src.constants import QnetType, RepeatedRolloutModelPath_10x10_4v4
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
    


class fullQnetBasedAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            qnet_name: str,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space

        self._nn = self._load_net(qnet_name)

    def _load_net(
            self,
            qnet_name: str = None
    ):
        net = ConvModel((4,self._grid_shape[0],self._grid_shape[1]),self._action_space.n)
        #net.load_state_dict(torch.load(qnet_name))
        net.load_state_dict(torch.load(qnet_name, map_location=torch.device('cpu')))

        # set dropout and batch normalization layers to evaluation mode
        net.eval()
        return net

    def act(
            self,
            obs: List[float],
            epsilon: float = 0.0,
            **kwargs,
    ) -> int:
        # 1) form 5 samples for each action
        # 2) call q-network
        # 3) arg max action OR random (epsilon greedy)
        p = np.random.random()
        if p < epsilon:
            # random action -> exploration
            return self._action_space.sample()
        else:
            qs = self._nn(torch.Tensor(obs).unsqueeze(0))
            return qs.max(-1)[-1].item()


class QnetBasedAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            m_agents: int,
            p_preys: int,
            grid_shape: Tuple[int, int],
            action_space: gym.spaces.Discrete,
    ):
        self.id = agent_id
        self._m_agents = m_agents
        self._p_preys = p_preys
        self._grid_shape = grid_shape
        self._action_space = action_space

        # load neural net on init
        # qnet_name = RolloutModelPath_10x10_4v2 if qnet_type == QnetType.BASELINE else RepeatedRolloutModelPath_10x10_4v2
        qnet_name = RepeatedRolloutModelPath_10x10_4v4

        self._nn = self._load_net(qnet_name)

    def act(
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
            epsilon: float = 0.0,
            **kwargs,
    ) -> int:
        # 1) form 5 samples for each action
        # 2) call q-network
        # 3) arg max action OR random (epsilon greedy)
        p = np.random.random()
        if p < epsilon:
            # random action -> exploration
            return self._action_space.sample()
        else:
            # argmax -> exploitation
            x = self._convert_to_x(obs, prev_actions)
            x = np.reshape(x, newshape=(1, -1))
            v = torch.from_numpy(x)
            qs = self._nn(v)
            return np.argmax(qs.data.numpy())


    def _load_net(
            self,
            qnet_name: str = None
    ) -> QNetworkCoordinated:
        net = QNetworkCoordinated(self._m_agents, self._p_preys, self._action_space.n)
        #net.load_state_dict(torch.load(qnet_name))
        net.load_state_dict(torch.load(qnet_name, map_location=torch.device('cpu')))

        # set dropout and batch normalization layers to evaluation mode
        net.eval()

        return net

    def _convert_to_x(
            self,
            obs: List[float],
            prev_actions: Dict[int, int] = None,
    ) -> np.ndarray:
        # state
        np_obs = np.array(obs, dtype=np.float32).flatten()

        # agent ohe
        agent_ohe = np.zeros(shape=(self._m_agents,), dtype=np.float32)
        agent_ohe[self.id] = 1.

        # prev actions
        prev_actions_ohe = np.zeros(shape=(self._m_agents * self._action_space.n,), dtype=np.float32)
        for agent_i, action_i in prev_actions.items():
            ohe_action_index = int(agent_i * self._action_space.n) + action_i
            prev_actions_ohe[ohe_action_index] = 1.

        # combine all
        x = np.concatenate((np_obs, agent_ohe, prev_actions_ohe))

        return x

if __name__ == "__main__":
    m = ConvModel((4,50,50),5)
    