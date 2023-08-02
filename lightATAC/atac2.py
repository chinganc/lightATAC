# yapf: disable
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightATAC.util import compute_batched, DEFAULT_DEVICE, update_exponential_moving_average, normalized_sum, torchify


def clamp(x, Vmin, Vmax): # clamp with gradient flow
    return x + (Vmax-x).clamp(max=0).detach() - (x-Vmin).clamp(max=0).detach()

class ATAC(nn.Module):
    """ Adversarilly Trained Actor Critic """
    def __init__(self, *,
                 policy,
                 qf,
                 target_qf=None,
                 optimizer,
                 discount=0.99,
                 Vmin=-float('inf'), # min value of Q (used in target backup)
                 Vmax=float('inf'), # max value of Q (used in target backup)
                 # Optimization parameters
                 actor_update_freq=100, # update the actor every n critic updates
                 policy_lr=5e-5,
                 qf_lr=5e-4,
                 target_update_tau=5e-3,
                 # ATAC parameters
                 beta=1.0,  # the regularization coefficient in front of the Bellman error
                 w1=0.5,  # weight on target error
                 w2=0.5,  # weight on residual error
                 # ATAC0 parameters
                 init_observations=None, # Provide it to use ATAC0 (None or np.ndarray)
                 buffer_batch_size=256,  # for ATAC0 (sampling batch_size of init_observations)
                 # Misc
                 debug=True,
                 ):

        #############################################################################################
        super().__init__()
        assert beta>=0
        policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.
        self._debug = debug  # log extra info

        # ATAC main parameter
        self.beta = beta # regularization constant on the Bellman surrogate

        # q update parameters
        self._discount = discount
        self._Vmin = Vmin  # lower bound on the target
        self._Vmax = Vmax  # upper bound on the target

        # norm constraint on the qf's weight; positive for l2; negative for l-inf
        self._w1 = w1/(w1+w2) # weight on target error
        self._w2 = w2/(w1+w2) # weight on residual error

        # networks
        self.policy = policy
        self._qf = qf
        self._target_qf = copy.deepcopy(self._qf).requires_grad_(False) if target_qf is None else target_qf

        # optimization
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._tau = target_update_tau
        self._actor_update_freq = actor_update_freq
        self._i = -1 # iteration counter

        self._optimizer = optimizer
        self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr) #  lr for warmstart
        self._qf_optimizer = self._optimizer(self._qf.parameters(), lr=self._qf_lr)

        # initial state pessimism (ATAC0)
        self._init_observations = torch.Tensor(init_observations) if init_observations is not None else init_observations  # if provided, it runs ATAC0
        self._buffer_batch_size = buffer_batch_size

    def update(self, observations, actions, next_observations, rewards, terminals, **kwargs):

        rewards = rewards.flatten()
        terminals = terminals.flatten().float()

        ##### Update Critic #####
        def compute_bellman_backup(q_pred_next):
            assert rewards.shape == q_pred_next.shape
            return clamp(rewards + (1.-terminals)*self._discount*q_pred_next, self._Vmin, self._Vmax)

        # Pre-computation
        with torch.no_grad():  # regression target
            new_next_actions = self.policy(next_observations).sample()
            target_q_values = self._target_qf(next_observations, new_next_actions)  # projection
            q_target = compute_bellman_backup(target_q_values.flatten())

        new_actions_dist = self.policy(observations)  # This will be used to compute the entropy
        new_actions = new_actions_dist.rsample() # These samples will be used for the actor update too, so they need to be traced.

        if self._init_observations is None:  #  relative pessimism (ATAC)
            pess_new_actions = new_actions.detach()
            pess_observations = observations
        else:  # absolute pessimism (PSPI)
            idx_ = np.random.choice(len(self._init_observations), min(self._buffer_batch_size, len(rewards)))
            init_observations = torchify(self._init_observations[idx_])
            init_actions_dist = self.policy(init_observations)
            pess_new_actions = init_actions_dist.rsample().detach()
            pess_observations = init_observations

        qf_pred_both, qf_pred_next_both, qf_new_actions_both \
            = compute_batched(self._qf.both, [observations, next_observations, pess_observations],
                                             [actions,      new_next_actions,  pess_new_actions])

        qf_loss = 0
        w1, w2 = self._w1, self._w2
        for i, (qfp, qfpn, qfna) in enumerate(zip(qf_pred_both, qf_pred_next_both, qf_new_actions_both)):
            # Compute Bellman error
            assert qfp.shape == qfpn.shape == qfna.shape == q_target.shape
            target_error = F.mse_loss(qfp, q_target)

            # Compute pessimism term
            if self._init_observations is None:  #  relative pessimism (ATAC)
                pess_loss = (qfna - qfp).mean()
            else:  # absolute pessimism (PSPI)
                pess_loss = qfna.mean()

            if i != len(qf_pred_both)-1:  # use td loss only
                pess_loss = pess_loss.detach()
                qf_bellman_loss = target_error
            else:  # use tdrs loss
                q_backup = compute_bellman_backup(qfpn)  # compared with `q_target``, the gradient of `self._qf` is traced in `q_backup`.
                residual_error = F.mse_loss(qfp, q_backup)
                qf_bellman_loss = w1*target_error+ w2*residual_error

            ## Compute full q loss (qf_loss = pess_loss + beta * qf_bellman_loss)
            qf_loss += normalized_sum(pess_loss, qf_bellman_loss, self.beta)

        # Update q
        self._qf_optimizer.zero_grad()
        qf_loss.backward()
        self._qf_optimizer.step()
        update_exponential_moving_average(self._target_qf, self._qf, self._tau)


        # Log
        log_info = dict(qf_loss=qf_loss.item(),
                        qf_bellman_loss=qf_bellman_loss.item(),
                        pess_loss=pess_loss.item())

        # For logging
        if self._debug:
            with torch.no_grad():
                debug_log_info = dict(
                        bellman_surrogate=residual_error.item(),
                        qf1_pred_mean=qf_pred_both[0].mean().item(),
                        qf2_pred_mean = qf_pred_both[1].mean().item(),
                        q_target_mean = q_target.mean().item(),
                        target_q_values_mean = target_q_values.mean().item(),
                        qf1_new_actions_mean = qf_new_actions_both[0].mean().item(),
                        qf2_new_actions_mean = qf_new_actions_both[1].mean().item(),
                        action_diff = torch.mean(torch.norm(actions - new_actions, dim=1)).item()
                        )
            log_info.update(debug_log_info)

        self._i += 1
        if not (self._i % self._actor_update_freq == 0):
            return log_info

        ##### Update Actor #####
        self._qf.requires_grad_(False)
        lower_bound = self._qf.both(observations, new_actions)[-1].mean() # just use one network
        self._qf.requires_grad_(True)
        policy_loss = -lower_bound

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # Log
        actor_log_info = dict(policy_loss=policy_loss.item(),
                              lower_bound=lower_bound.item(),
                              policy_grad=sum([ x.grad.norm() for x in self.policy.parameters()]).item())
        log_info.update(actor_log_info)

        return log_info