import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, TransformedDistribution, Normal
from torch.distributions.transforms import TanhTransform, AffineTransform
import numpy as np
from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

# Below are modified from gwthomas/IQL-PyTorch

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2,
                init_std=1.0, use_tanh=False, min_std=1e-5, max_std=10, std_type='constant',
                action_scale=1.0, activation=nn.ReLU):
        super().__init__()
        init_log_std = np.log(init_std)
        self.std_type = std_type
        if self.std_type=='diagonal':
            # the first half of output predicts the mean; the second half predicts log_std
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim*2], activation=activation)
            self.net[-1].weight.data[act_dim:] *= 0.
            self.net[-1].bias.data[act_dim:] = init_log_std
        elif self.std_type=='diagonal_detached':
            # the first half of output predicts the mean; the second half predicts log_std
            self.feature = mlp([obs_dim, *([hidden_dim] * (n_hidden-1)), hidden_dim], activation=activation, output_activation=activation)
            self.mean_net =  nn.Sequential(nn.Linear(hidden_dim, act_dim))
            self.log_std_net =  nn.Sequential(nn.Linear(hidden_dim, act_dim))
            self.log_std_net[-1].weight.data[:] *= 0.
            self.log_std_net[-1].bias.data[:] = init_log_std
        elif self.std_type=='separate':
            # the first half of output predicts the mean; the second half predicts log_std
            self.mean_net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim], activation=activation)
            self.log_std_net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim], activation=activation)
            self.log_std_net[-1].weight.data[:] *= 0.
            self.log_std_net[-1].bias.data[:] = init_log_std
        elif self.std_type=='constant':
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim], activation=activation)
            self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32)* init_log_std)
        else:
            raise ValueError
        self.use_tanh = use_tanh
        self.min_log_std = np.log(min_std)
        self.max_log_std = np.log(max_std)
        self.action_scale = action_scale

    def forward(self, obs, ignore_transform=False):
        if self.std_type=='diagonal':
            out = self.net(obs)
            mean, log_std = out.split(out.shape[-1]//2, dim=-1)
            std = torch.exp(log_std.clamp(self.min_log_std, self.max_log_std))
            dist = Normal(mean, std)
        elif self.std_type=='diagonal_detached':
            feature = self.feature(obs)
            mean = self.mean_net(feature)
            log_std = self.log_std_net(feature.detach())
            std = torch.exp(log_std.clamp(self.min_log_std, self.max_log_std))
            dist = Normal(mean, std)
        elif self.std_type=='separate':
            mean, log_std = self.mean_net(obs), self.log_std_net(obs)
            std = torch.exp(log_std.clamp(self.min_log_std, self.max_log_std))
            dist = Normal(mean, std)
        elif self.std_type=='constant':
            mean = self.net(obs)
            std = torch.exp(self.log_std.clamp(self.min_log_std, self.max_log_std))
            scale_tril = torch.diag(std)
            dist = MultivariateNormal(mean, scale_tril=scale_tril)
        else:
            raise ValueError
        if self.use_tanh and not ignore_transform:
            dist = TransformedDistribution(dist, TanhTransform(cache_size=1))
        if not ignore_transform and hasattr(self, 'action_scale'):  # backward compatibility
            # apply scaling; this is mainly for addressing the numerical issue of tanh
            dist = TransformedDistribution(dist,
                    AffineTransform(0.0, self.action_scale, event_dim=1, cache_size=1))
        return dist

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs, ignore_transform=True)  # just Gaussian
            act = dist.mean if deterministic else dist.sample()
            if self.use_tanh:
                act = torch.tanh(act)
            if hasattr(self, 'action_scale'):
                act *= self.action_scale
            return act


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2, use_tanh=False):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)
        self.use_tanh = use_tanh

    def forward(self, obs):
        return torch.tanh(self.net(obs)) if self.use_tanh else self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)

import torch.nn.functional as F
from torch.distributions.categorical import Categorical
class Probabilty:  # TODO inherit from torch.distributions.Distribution?
    #  This is a dummy class.
    def __init__(self, logits):
        self.logits = logits
    def rsample(self):
        return F.softmax(self.logits, dim=-1)
        # return Categorical(logits=self.logits).sample()
    def sample(self):
        return self.rsample().detach()
    def log_prob(self, x):
        return -F.cross_entropy(self.logits, x, reduction='none')
    # def prob(self):
    #     return F.softmax(self.logits, dim=-1)

class SoftmaxPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_bins=None, hidden_dim=256, n_hidden=2, activation=nn.ReLU):
        super().__init__()
        if n_bins is None:
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim], activation=activation)
        else:
            self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim*n_bins], activation=activation)
        self.act_dim = act_dim
        self.n_bins = n_bins
    def forward(self, obs):
        out = self.net(obs)
        if self.n_bins is not None:
            # out: batch x act_dim*n_bins
            # compute outer sum and reshape it to batch x n_bins ** act_dim
            batch_size = out.shape[0]
            out = out.reshape((-1, self.act_dim, self.n_bins))
            new_out = out[:,0]
            for i in range(1,self.act_dim):
                new_out = new_out.unsqueeze(-1) + out[:,i].unsqueeze(1)
                new_out = new_out.reshape([batch_size, -1])
            out = new_out
        return Probabilty(out)
    def act(self, obs, deterministic=False, enable_grad=False):
        return Categorical(logits=self.net(obs)).sample()