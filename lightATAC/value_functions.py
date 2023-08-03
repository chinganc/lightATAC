import torch
import torch.nn as nn
from .util import mlp

# Below are modified from gwthomas/IQL-PyTorch

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, ignore_actions=False):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)
        self.ignore_actions = ignore_actions

    def both(self, state, action, index=None):
        if self.ignore_actions:
            action = action * 0
        sa = torch.cat([state, action], 1)
        if index==0: # q1
            return self.q1(sa)
        elif index==1: # q2
            return self.q2(sa)
        else:
            return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)