import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import mlp

# Below are modified from gwthomas/IQL-PyTorch

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self._q1 = mlp(dims, squeeze_output=True)
        self._q2 = mlp(dims, squeeze_output=True)

    def q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self._q1(sa)

    def q2(self, state, action):
        sa = torch.cat([state, action], 1)
        return self._q2(sa)

    def both(self, state, action, index=None):
        return self.q1(state, action), self.q2(state, action)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))

class TwinQDiscrete(TwinQ):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        nn.Module.__init__(self)
        dims = [state_dim, *([hidden_dim] * n_hidden), action_dim]
        self._q1 = mlp(dims)
        self._q2 = mlp(dims)

    def q1(self, state, action):
        action = self._convert_action(action)
        return torch.sum(self._q1(state) * action, dim=-1)

    def q2(self, state, action):
        action = self._convert_action(action)
        return torch.sum(self._q2(state) * action, dim=-1)

    def _convert_action(self, action):
        # convert action from int to one-hot when needed
        if action.dtype == torch.int64 and len(action.shape) == 1:
            action = F.one_hot(action, self._q1[-1].out_features)
        return action

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)