import numpy as np
import gym
import torch
import torch.nn as nn
from lightATAC.util import DEFAULT_DEVICE, mlp
from lightATAC.value_functions import TwinQ
from lightATAC.policy import Categorical, Probabilty

def c2d(x, min=-1, max=1, n_bins=10):
    """ Convert a continuous vector to discrete (int) by binning each dimension. """
    assert len(x.shape)==2  # batch x dim
    digit = (((x-min)/(max-min))*(n_bins-1)).astype(np.int64)
    return np.sum(digit * (n_bins**np.arange(digit.shape[-1])), axis=1)

def d2c(x, dim, min=-1, max=1, n_bins=10):
    """ Reverse of c2d. """
    if len(x.shape)==0:  # add a batch dim
        x = x[np.newaxis]
    if len(x.shape)==1:
        x = x[:,np.newaxis]
    assert len(x.shape)==2 and x.shape[1]==1  # x should be batch x 1
    x = x // np.repeat(n_bins**np.arange(dim)[np.newaxis,...], len(x), axis=0) % n_bins
    return (x.astype(np.float64) /(n_bins-1))*(max-min)+min

def torchc2d(x, min=-1, max=1, n_bins=10):
    """ Convert a continuous vector to discrete (int) by binning each dimension. """
    assert len(x.shape)==2  # batch x dim
    dim = x.shape[-1]
    digit = (((x-min)/(max-min))*(n_bins-1)).long()
    return torch.sum(digit * (n_bins**torch.arange(dim)), axis=1)

def torchd2c(x, dim, min=-1, max=1, n_bins=10):
    """ Reverse of c2d. """
    if len(x.shape)==0:  # add a batch dim
        x = x[None]
    if len(x.shape)==1:
        x = x[:,None]
    assert len(x.shape)==2 and x.shape[1]==1  # x should be batch x 1
    x = torch.div(x, (n_bins**torch.arange(dim).to(x.device)[None,...]).repeat(len(x),1), rounding_mode='floor') % n_bins
    return (x.float()/(n_bins-1))*(max-min)+min

class DiscreteActionGymWrapper(gym.Wrapper):
    def __init__(self, env, d2c):
        super().__init__(env)
        self._d2c = d2c
    def step(self, action):
        action = self._d2c(action)
        return super().step(action)

class C2DSoftmaxPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_bins, independent=False, hidden_dim=256, n_hidden=2, activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim*n_bins], activation=activation)
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.n_bins = n_bins

    def forward(self, obs):
        if len(obs.shape)<2:
            # add batch dimension
            assert obs.shape[0]==self.obs_dim
            obs = obs.unsqueeze(0)

        out = self.net(obs)  # out: batch x act_dim*n_bins

        # compute outer sum and reshape it to batch x n_bins ** act_dim
        batch_size = out.shape[0]
        out = out.reshape((-1, self.act_dim, self.n_bins))
        new_out = out[:,0]
        for i in range(1,self.act_dim):
            new_out = new_out.unsqueeze(-1) + out[:,i].unsqueeze(1)
            new_out = new_out.reshape([batch_size, -1])
        out = new_out
        return Probabilty(out.squeeze())

    def act(self, obs, deterministic=False, enable_grad=False):
        return Categorical(logits=self(obs).logits).sample()


class D2CTwinQWrapper(TwinQ):
    # TODO should be a wrapper
    def __init__(self, twinq, n_bins, action_dim, min=-1, max=1, hidden_dim=256, n_hidden=2):
        torch.nn.Module.__init__(self)
        self.twinq = twinq
        self._n_bins = n_bins
        self._min = min
        self._max = max
        self._action_dim = action_dim
        self._n_tokens = n_bins**action_dim
        self._action_basis =  torchd2c(torch.arange(self._n_tokens, device=DEFAULT_DEVICE),dim=action_dim, min=self._min, max=self._max, n_bins=n_bins)[None]  # 1 x n_tokens x action_dim

    def _q_wrapper(self, qf, state, action, centered=True):
        # receive discrete action (one-hot/distribution)
        # state: batch_size x state_dim
        # action: batch_size x n_tokens
        batch_size = len(state)
        dim = self._action_dim
        n_tokens = self._n_tokens
        n_bins = self._n_bins
        assert n_tokens==action.shape[-1]

        # Create action basis
        ind = Categorical(probs=action).sample() # batch_size
        prob = action[torch.arange(batch_size), ind]
        action_basis = self._action_basis[0][ind]  # batch_size x action_dim
        q_all = qf(state, action_basis) # batch_size
        cv = q_all.mean().detach() if centered else 0.
        q = prob/prob.detach()* (q_all-cv) + cv # batch_size NOTE This only affects the gradient, not the value.

        # NOTE This queries q for all the actions, which is too slow.
        # action_basis = self._action_basis.repeat(batch_size, 1, 1) # batch_size x n_tokens  x action_dim
        # action_extended = action_basis.reshape(-1, dim)  # batch_size*n_tokens  x action_dim
        # state_extended = state.repeat_interleave(n_tokens, dim=0)  # batch_size*n_tokens x state_dim
        # q_all = qf(state_extended, action_extended).reshape(batch_size,n_tokens)  # batch_size x n_tokens
        # assert q_all.shape == action.shape
        # q = (q_all * action).sum(dim=-1)  # batch_size
        return q

    def q1(self, state, action):
        action = self._convert_action(action)
        return self._q_wrapper(self.twinq.q1, state, action)

    def q2(self, state, action):
        action = self._convert_action(action)
        return self._q_wrapper(self.twinq.q2, state, action)

    def _convert_action(self, action):
        # convert action from int to one-hot when needed
        if action.dtype == torch.int64 and len(action.shape) == 1:
            action = F.one_hot(action, self._n_tokens)
        return action

# class D2CTwinQ(TwinQ):
#     # TODO should be a wrapper
#     def __init__(self, state_dim, action_dim, n_bins, min=-1, max=1, hidden_dim=256, n_hidden=2):
#         super().__init__(state_dim, action_dim, hidden_dim=hidden_dim, n_hidden=n_hidden)
#         self._n_bins = n_bins
#         self._min = min
#         self._max = max
#         self._action_dim = action_dim
#         self._n_tokens = n_bins**action_dim
#         self._action_basis =  torchd2c(torch.arange(self._n_tokens, device=DEFAULT_DEVICE),dim=action_dim, min=self._min, max=self._max, n_bins=n_bins)[None]  # 1 x n_tokens x action_dim

#     def _q_wrapper(self, qf, state, action, centered=True):
#         # receive discrete action (one-hot/distribution)
#         # state: batch_size x state_dim
#         # action: batch_size x n_tokens
#         batch_size = len(state)
#         dim = self._action_dim
#         n_tokens = self._n_tokens
#         n_bins = self._n_bins
#         assert n_tokens==action.shape[-1]

#         # Create action basis
#         ind = Categorical(probs=action).sample() # batch_size
#         prob = action[torch.arange(batch_size), ind]
#         action_basis = self._action_basis[0][ind]  # batch_size x action_dim
#         q_all = qf(state, action_basis) # batch_size
#         cv = q_all.mean().detach() if centered else 0.
#         q = prob/prob.detach()* (q_all-cv) + cv # batch_size NOTE This only affects the gradient, not the value.

#         # NOTE This queries q for all the actions, which is too slow.
#         # action_basis = self._action_basis.repeat(batch_size, 1, 1) # batch_size x n_tokens  x action_dim
#         # action_extended = action_basis.reshape(-1, dim)  # batch_size*n_tokens  x action_dim
#         # state_extended = state.repeat_interleave(n_tokens, dim=0)  # batch_size*n_tokens x state_dim
#         # q_all = qf(state_extended, action_extended).reshape(batch_size,n_tokens)  # batch_size x n_tokens
#         # assert q_all.shape == action.shape
#         # q = (q_all * action).sum(dim=-1)  # batch_size
#         return q

#     def q1(self, state, action):
#         action = self._convert_action(action)
#         return self._q_wrapper(super().q1, state, action)

#     def q2(self, state, action):
#         action = self._convert_action(action)
#         return self._q_wrapper(super().q2, state, action)

#     def _convert_action(self, action):
#         # convert action from int to one-hot when needed
#         if action.dtype == torch.int64 and len(action.shape) == 1:
#             action = F.one_hot(action, self._n_tokens)
#         return action
