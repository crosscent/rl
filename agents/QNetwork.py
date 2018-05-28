import math
import random
from collections import deque
from typing import NamedTuple, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor


class Transition(NamedTuple):
    state: Tuple[float, ...]
    action: int
    reward: float
    next_state: Tuple[float, ...]


class ExperienceReplay:
    def __init__(self, memory_size):
        self._memory = deque(maxlen=memory_size)

    def add(self, transition: Transition):
        self._memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._memory, min(len(self._memory), batch_size))

    @property
    def size(self):
        return len(self._memory)


class Network(nn.Module):
    def __init__(self, dim_actions: int, dim_states: int):
        super(Network, self).__init__()
        self.input = nn.Linear(dim_states, 24)
        self.hidden1 = nn.Linear(24, 48)
        self.output = nn.Linear(48, dim_actions)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden1(x)
        x = F.tanh(x)
        return self.output(x)


class QNetwork:
    def __init__(self,
                 dim_actions: int,
                 dim_states: int,
                 replay_size: int = 10000,
                 batch_size: int = 500,
                 learning_rate: float = 0.001,
                 gamma=0.9):
        # assign input dimensions
        self._dim_actions = dim_actions
        self._dim_states = dim_states

        self._step = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._experience_replay = ExperienceReplay(replay_size)
        self._batch_size = batch_size
        self._gamma = gamma

        self._policy_network = Network(self._dim_actions, self._dim_states).to(self._device)
        self._target_network = Network(self._dim_actions, self._dim_states).to(self._device)
        self.update_target()

        self._optimizer = optim.Adam(self._policy_network.parameters(), lr=learning_rate, amsgrad=True)

    def get_action(self, state: np.ndarray):
        self._step += 1
        if self._explore():
            return torch.tensor([random.randrange(self._dim_actions)], dtype=torch.long)
        else:
            with torch.no_grad():
                return self._policy_network(self._parse_state(state)).argmax()

    def learn(self):
        if self._experience_replay.size < self._batch_size:
            return
        else:
            self._optimize()

    def update_target(self):
        self._target_network.load_state_dict(self._policy_network.state_dict())

    def add_history(self, s, a, r, s_):
        state = self._parse_state(s)
        action = torch.tensor([[a]], device=self._device)
        reward = torch.tensor([r], device=self._device)
        next_state = self._parse_state(s_) if s_ is not None else None

        transition = Transition(state=state, action=action, reward=reward, next_state=next_state)
        self._experience_replay.add(transition)

    def _optimize(self):
        transitions = self._experience_replay.sample(self._batch_size)
        states, actions, rewards, next_states = zip(*transitions)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)

        # compute Q(s, a)* = r + gamma * argmax_a Q(s', a)
        state_action_values = self._policy_network(states).gather(1, actions)

        non_terminating_mask = torch.tensor([next_state is not None for next_state in next_states],
                                            device=self._device, dtype=torch.uint8)
        non_terminating_next_states = torch.cat([next_state for next_state in next_states if next_state is not None])

        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_terminating_mask] = self._target_network(non_terminating_next_states).max(1)[0].detach()

        expected_state_values = rewards + (next_state_values * self._gamma)

        loss = F.mse_loss(state_action_values, expected_state_values.unsqueeze(1))

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _explore(self) -> bool:
        esp_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1 * self._step / 1000)
        return random.random() < esp_threshold

    def _parse_state(self, state: np.ndarray) -> Tensor:
        return torch.from_numpy(state).float().unsqueeze(0).to(self._device)
