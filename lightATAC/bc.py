
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from copy import deepcopy
from lightATAC.util import compute_batched, discount_cumsum, sample_batch, \
        traj_data_to_qlearning_data, tuple_to_traj_data, update_exponential_moving_average


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class BehaviorPretraining(nn.Module):
    """
        A generic pretraining algorithm for learning the behavior policy and its values.

        It trains the policy by behavior cloning (MLE or L2 error), and the
        values (v and q) by TD-lambda and expectile regression (by default, it
        uses least squares.)

    """

    def __init__(self, *,
                 policy=None,  # nn.module
                 qf=None,  # nn.module
                 vf=None,  # nn.module
                 discount=0.99,  # discount factor
                 lambd=1.0,  # lambda for TD lambda
                 td_weight=1.0,  # weight on the td error (surrogate based on target network)
                 rs_weight=0.0,  # weight on the residual error
                 lr=5e-4,  # learning rate
                 target_update_rate=5e-3,
                 expectile=0.5):

        super().__init__()
        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.discount = discount
        self.lambd = lambd
        self.td_weight = td_weight / (td_weight+rs_weight)
        self.rs_weight = rs_weight / (td_weight+rs_weight)
        self.target_update_rate = target_update_rate
        self.expectile = expectile

        if self.qf is not None:
            assert self.policy is not None, 'Learning a q network requires a policy network.'
            self.target_qf = deepcopy(self.qf).requires_grad_(False)

        parameters = sum([ list(x.parameters()) for x in (policy, qf, vf) if x is not None], start=[])
        self.optimizer = torch.optim.Adam(parameters, lr=lr)

    def train(self, dataset, n_steps, batch_size=256, log_freq=1000, log_fun=None, silence=False):
        """ A basic trainer loop. Users cand customize this method if needed.

            dataset: a dict of observations, actions, rewards, terminals
        """
        traj_data = tuple_to_traj_data(dataset)
        if self.vf is not None:
            self.preprocess_traj_data(traj_data, self.discount)
        dataset = traj_data_to_qlearning_data(traj_data)  # make sure `next_observations` is there

        for step in trange(n_steps, disable=silence):
            train_metrics = self.update(**sample_batch(dataset, batch_size))
            if (step % max(log_freq,1) == 0 or step==n_steps-1) and log_fun is not None:
                log_fun(train_metrics, step)
        return dataset

    def compute_qf_loss(self, observations, actions, next_observations, rewards, terminals, **kwargs):
        # Q Loss with TD error and/or Residual Error using Target Q
        def compute_bellman_backup(v_next):
            assert rewards.shape == v_next.shape
            return rewards + (1.-terminals.float())*self.discount*v_next
        qf_loss = 0.
        # Update target
        update_exponential_moving_average(self.target_qf, self.qf, self.target_update_rate)
        # Compute shared parts
        qs_all = self.qf.both(observations, actions)  # tuple, inference
        with torch.no_grad():
            next_policy_outs = self.policy(next_observations)   # inference
            if isinstance(next_policy_outs, torch.distributions.Distribution):
                next_policy_actions = next_policy_outs.sample()
            else:
                next_policy_actions = next_policy_outs
        # Temporal difference error
        if self.td_weight>0:
            next_targets = self.target_qf(next_observations, next_policy_actions)  # inference
            td_targets = compute_bellman_backup(next_targets)
            for qs in qs_all:
                qf_loss += asymmetric_l2_loss(td_targets - qs, self.expectile) * self.td_weight
        # Residual error
        if self.rs_weight>0:
            next_qs_all = self.qf.both(next_observations, next_policy_actions)  # inference
            for qs, next_qs in zip(qs_all, next_qs_all):
                rs_targets = compute_bellman_backup(next_qs)
                qf_loss += asymmetric_l2_loss(rs_targets - qs, self.expectile) * self.rs_weight
        # Log
        info_dict = {"Q loss": qf_loss.item(),
                     "Average Q value": qs.mean().item()}
        return qf_loss, info_dict

    def compute_vf_loss(self, observations, actions, next_observations, rewards, terminals,
                      returns, remaining_steps, last_observations, last_terminals, **kwargs):
        # V loss (TD-lambda)

        # Monte-Carlo Q estimate
        mc_estimates = torch.zeros_like(rewards)
        if self.lambd>0:
            last_vs = self.vf(last_observations)  # inference
            mc_estimates = returns + (1-last_terminals.float()) * self.discount**remaining_steps * last_vs
        if self.lambd<1:
            with torch.no_grad():
                v_next = self.vf(next_observations)  # inference
                td_targets = rewards + (1.-terminals.float())*self.discount*v_next
        vs = self.vf(observations)  # inference
        # TD error
        td_error = 0.
        vf_loss = asymmetric_l2_loss(td_targets - vs, self.expectile)
        # Log
        info_dict = {"V loss": vf_loss.item(),
                     "Average V value": vs.mean().item()}
        return vf_loss, info_dict

    def compute_policy_loss(self, observations, actions, **kwargs):
        # Policy loss
        policy_outs = self.policy(observations)
        if isinstance(policy_outs, torch.distributions.Distribution):  # MLE
            bc_losses = -policy_outs.log_prob(actions)
        elif torch.is_tensor(policy_outs):  # l2 loss
            assert policy_outs.shape == actions.shape
            bc_losses = torch.sum((policy_outs - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(bc_losses)
        info_dict = {"Policy loss": policy_loss.item()}
        return policy_loss, info_dict

    def update(self, **batch):
        qf_loss = vf_loss = policy_loss = torch.tensor(0., device=batch['observations'].device)
        qf_info_dict = vf_info_dict = policy_info_dict = {}
        # Compute loss
        if self.qf is not None:
            qf_loss, qf_info_dict = self.compute_qf_loss(**batch)
        if self.vf is not None:
            vf_loss, vf_info_dict = self.compute_vf_loss(**batch)
        if self.policy is not None:
            policy_loss, policy_info_dict = self.compute_policy_loss(**batch)
        # Update
        loss = policy_loss + qf_loss + vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Log
        info_dict = {**qf_info_dict, **vf_info_dict, **policy_info_dict}
        return info_dict

    @classmethod
    def preprocess_traj_data(cls, traj_data, discount):
        for traj in traj_data:
            H = len(traj['rewards'])
            if torch.is_tensor(traj['rewards']):
                with torch.no_grad():
                    traj['returns'] = discount_cumsum(traj['rewards'], discount)
                    assert traj['returns'].shape == traj['rewards'].shape
                    traj['remaining_steps'] = torch.flip(torch.arange(H, device=traj['rewards'].device), dims=(0,))+1
                    assert traj['remaining_steps'].shape == traj['rewards'].shape
                    traj['last_observations'] = torch.repeat_interleave(traj['observations'][-1:], H, dim=0)
                    assert traj['last_observations'].shape ==traj['observations'].shape
                    traj['last_terminals'] = torch.repeat_interleave(traj['terminals'][-1], H)
                    assert traj['last_terminals'].shape == traj['terminals'].shape
            else:
                traj['returns'] = discount_cumsum(traj['rewards'], discount)
                assert traj['returns'].shape == traj['rewards'].shape
                traj['remaining_steps'] = np.flip(np.arange(H))+1
                assert traj['remaining_steps'].shape == traj['rewards'].shape
                traj['last_observations'] = np.repeat(traj['observations'][-1:], H, axis=0)
                assert traj['last_observations'].shape ==traj['observations'].shape
                traj['last_terminals'] = np.repeat(traj['terminals'][-1], H)
                assert traj['last_terminals'].shape == traj['terminals'].shape
