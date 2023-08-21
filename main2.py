from pathlib import Path
import gym, d4rl
import numpy as np
import torch, copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from lightATAC.policy import GaussianPolicy, GMMPolicy
from lightATAC.value_functions import TwinQ, ValueFunction
from lightATAC.util import Log, set_seed
from lightATAC.bp import BehaviorPretraining
from lightATAC.atac2 import ATAC
from lightATAC.util import evaluate_policy, sample_batch, traj_data_to_qlearning_data, tuple_to_traj_data, DEFAULT_DEVICE

EPS=1e-6

def eval_agent(*, env, agent, discount, n_eval_episodes, max_episode_steps=1000,
               deterministic_eval=True, normalize_score=None):

    all_returns = np.array([evaluate_policy(env, agent, max_episode_steps, deterministic_eval, discount) \
                             for _ in range(n_eval_episodes)])
    eval_returns = all_returns[:,0]
    discount_returns = all_returns[:,1]

    info_dict = {
        "return mean": eval_returns.mean(),
        "return std": eval_returns.std(),
        "discounted returns": discount_returns.mean()
    }

    if normalize_score is not None:
        normalized_returns = normalize_score(eval_returns)
        info_dict["normalized return mean"] = normalized_returns.mean()
        info_dict["normalized return std"] =  normalized_returns.std()
    return info_dict

def get_dataset(env):
    from urllib.error import HTTPError
    while True:
        try:
            return env.get_dataset()
        except (HTTPError, OSError):
            print('Unable to download dataset. Retry.')

def get_env_and_dataset(env_name):
    env = gym.make(env_name)  # d4rl ENV
    dataset = get_dataset(env)
    # process rewards such that V(done)=0 is correct.
    if  env_name in ('kitchen-complete-v0', 'kitchen-partial-v0', 'kitchen-mixed-v0'):
        assert len(env.TASK_ELEMENTS) >= dataset['rewards'].max()
        assert env.TERMINATE_ON_TASK_COMPLETE
        dataset['rewards'] -= len(env.TASK_ELEMENTS)
        # fix inconsistent terminals
        traj_data = tuple_to_traj_data(dataset)
        for traj in traj_data:
            traj['terminals'] = traj['rewards']==0
            traj['timeouts'] = np.zeros_like(traj['timeouts'], dtype=bool)
            traj['timeouts'][-1] = not traj['terminals'][-1]
        dataset = traj_data_to_qlearning_data(traj_data)
    return env, dataset

def main(args):
    # ------------------ Initialization ------------------ #
    torch.set_num_threads(1)
    env, dataset = get_env_and_dataset(args.env_name)
    set_seed(args.seed, env=env)

    # Scale and shift rewards
    dataset['rewards'] = (dataset['rewards'] + args.reward_bias) * args.reward_scale
    # XXX TODO Reward normalization.
    # To make sure the r_min is bounded away from zero, so that a randomly
    # initialized Q function is pessimistic.
    r_min = dataset['rewards'].min()
    if args.normalize_reward and  r_min <1.0:
        terminal_ind = dataset['terminals']>0
        shift = 1.0-r_min  # shift r_min to be 1.0
        dataset['rewards'][~terminal_ind] += shift
        dataset['rewards'][terminal_ind] += shift/(1-args.discount)
    print(f"r_min: {dataset['rewards'].min()}, r_max: {dataset['rewards'].max()}")
     # end of XXX

    # Set range of value functions
    Vmax, Vmin = float('inf'), -float('inf')
    if args.clip_v:
        Vmax = max(0.0, dataset['rewards'].max()/(1-args.discount))
        Vmin = min(0.0, dataset['rewards'].min()/(1-args.discount), Vmax-1.0/(1-args.discount))

    # Setup logger
    log_path = Path(args.log_dir) / args.env_name / ('_beta' + str(args.beta) + '_n_modes' + str(args.n_modes) + '_' + args.note)
    log = Log(log_path, vars(args))
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)

    # Assume vector observation and action
    obs_dim, act_dim = dataset['observations'].shape[1], dataset['actions'].shape[1]
    qf = TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, aggregation=args.aggregation).to(DEFAULT_DEVICE)
    target_qf = copy.deepcopy(qf).requires_grad_(False)
    if args.n_modes == 0:
        policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                                use_tanh=True, std_type='diagonal').to(DEFAULT_DEVICE)
    else:
        policy = GMMPolicy(obs_dim, act_dim,
                           n_modes=args.n_modes,
                           hidden_dim=args.hidden_dim,
                           n_hidden=args.n_hidden,
                           use_tanh=True,
                           action_scale=args.action_scale,
                           std_type='diagonal').to(DEFAULT_DEVICE)
    dataset['actions'] = np.clip(dataset['actions'], -1+EPS, 1-EPS)  # due to tanh
    rl = ATAC(
        policy=policy,
        qf=qf,
        target_qf=target_qf,
        optimizer=torch.optim.Adam,
        discount=args.discount,
        buffer_batch_size=args.batch_size,
        actor_update_freq=args.actor_update_freq,
        policy_lr=args.slow_lr,
        qf_lr=args.fast_lr,
        # ATAC main parameters
        beta=args.beta, # the regularization coefficient in front of the Bellman error
        Vmin=Vmin,
        Vmax=Vmax,
    ).to(DEFAULT_DEVICE)

    # ------------------ Pretraining ------------------ #
    # Train policy and value to fit the behavior data
    bp = BehaviorPretraining(policy=policy, lr=args.fast_lr).to(DEFAULT_DEVICE)
    def bp_log_fun(metrics, step):
        print(step, metrics)
        for k, v in metrics.items():
            writer.add_scalar('BehaviorPretraining/'+k, v, step)

    # XXX
    if args.load_pretrained:
        ## Cache pretrained model
        import os
        if not os.path.exists('pretrained_model'): # TODO: rename
            os.makedirs('pretrained_model')
        filename = 'pretrained_model/'+args.env_name+'_'+str(args.n_modes)+'_'+str(args.n_hidden)+'_pretrained_model.pt'
        if os.path.exists(filename):
            print('Load pretrained model from disk.')
            bp.load_state_dict(torch.load(filename))
            args.n_warmstart_steps = 0
            dataset = bp.train(dataset, args.n_warmstart_steps, log_fun=bp_log_fun, silence=args.disable_tqdm)  # This ensures "next_observations" is in `dataset`.
        else:
            dataset = bp.train(dataset, args.n_warmstart_steps, log_fun=bp_log_fun, silence=args.disable_tqdm)  # This ensures "next_observations" is in `dataset`.
            torch.save(bp.state_dict(),filename)
        ##
    else:
        dataset = bp.train(dataset, args.n_warmstart_steps, log_fun=bp_log_fun, silence=args.disable_tqdm)  # This ensures "next_observations" is in `dataset`.
    # end of XXX

    # Main Training

    # XXX Extract trajectories with done
    traj_data = tuple_to_traj_data(dataset)
    traj_data_w_done = [ t for t in traj_data if t['terminals'].sum()>0 ]
    dataset_w_done = traj_data_to_qlearning_data(traj_data_w_done)
    print('Number of trajectories:', len(traj_data), 'Number of transitions', len(dataset['observations']))
    print('Number of trajectories with done:', len(traj_data_w_done), 'Number of transitions with done:', len(dataset_w_done['observations']))
     # end of XXX

    for step in trange(args.n_steps, disable=args.disable_tqdm):
        # TODO clean this up
        # XXX
        batch = sample_batch(dataset, args.batch_size)
        batch['log'] = torch.zeros_like(batch['rewards'], dtype=torch.bool, device=DEFAULT_DEVICE)
        batch_done = sample_batch(dataset_w_done, args.done_batch_size)
        batch_done['log'] = torch.ones_like(batch_done['rewards'], dtype=torch.bool, device=DEFAULT_DEVICE)
        full_batch = {k: torch.cat([batch[k], batch_done[k]], dim=0) for k in batch.keys()}
         # end of XXX

        # Update
        train_metrics = rl.update(**full_batch)

        # train_metrics = rl.update(**sample_batch(dataset, args.batch_size))

        if step % max(int(args.eval_period/10),1) == 0  or  step==args.n_steps-1:
            print(train_metrics)
            for k, v in train_metrics.items():
                writer.add_scalar('Train/'+k, v, step)
        if step % args.eval_period == 0:
            eval_metrics = eval_agent(env=env,
                                      agent=policy,
                                      discount=args.discount,
                                      deterministic_eval=(args.n_modes == 0), # should not use determinsitic eval for gmm policies
                                      n_eval_episodes=args.n_eval_episodes,
                                      normalize_score=lambda returns: d4rl.get_normalized_score(args.env_name, returns)*100.0,
                                      deterministic_eval=args.deterministic_eval)
            log.row(eval_metrics)
            for k, v in eval_metrics.items():
                writer.add_scalar('Eval/'+k, v, step)
    # Final processing
    torch.save(rl.state_dict(), log.dir/'final.pt')
    log.close()
    writer.close()
    return eval_metrics['normalized return mean']



def get_parser():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--n_modes', type=int, default=0)
    parser.add_argument('--n_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--fast_lr', type=float, default=5e-4)
    parser.add_argument('--slow_lr', type=float, default=5e-5)
    parser.add_argument('--actor_update_freq', type=int, default=100)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--eval_period', type=int, default=5000)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--n_warmstart_steps', type=int, default=100*10**3)
    parser.add_argument('--clip_v', action='store_true')
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--action_scale', type=float, default=1.0)
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--reward_scale', type=float, default=1.0)
    parser.add_argument('--reward_bias', type=float, default=0.0)
    parser.add_argument('--normalize_reward', action='store_true')
    parser.add_argument('--done_batch_size', type=int, default=256)
    parser.add_argument('--deterministic_eval', action='store_true')
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--aggregation', type=str, default='min')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
