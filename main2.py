from pathlib import Path
import gym, d4rl
import numpy as np
import torch, copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from lightATAC.policy import GaussianPolicy, SoftmaxPolicy
from lightATAC.value_functions import TwinQ, TwinQDiscrete
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
    return env, dataset


def c2d(x, min=-1, max=1, n=10):
    assert len(x.shape)==2  # batch x dim
    digit = (((x-min)/(max-min))*(n-1)).astype(int)
    return np.sum(digit * (n**np.arange(digit.shape[-1])), axis=1)

def d2c(x, dim, min=-1, max=1, n=10):
    # reverse of c2d
    if len(x.shape)==0:  # add a batch dim
        x = x[np.newaxis]
    if len(x.shape)==1:
        x = x[:,np.newaxis]
    x = x // np.repeat(n**np.arange(dim)[np.newaxis,...], len(x), axis=0) % n
    return (x/(n-1))*(max-min)+min
class DiscreteActionGymWrapper(gym.Wrapper):
    def __init__(self, env, dim, min=-1, max=1, n=10):
        super().__init__(env)
        self._min = min
        self._max = max
        self._n = n
        self._dim = dim
    def step(self, action):
        action = d2c(action, dim=self._dim, min=self._min, max=self._max, n=self._n)[0]
        return super().step(action)

def main(args):
    # ------------------ Initialization ------------------ #
    torch.set_num_threads(1)
    env, dataset = get_env_and_dataset(args.env_name)
    set_seed(args.seed, env=env)

    # Set range of value functions
    Vmax, Vmin = float('inf'), -float('inf')
    if args.clip_v:
        Vmax = max(0.0, dataset['rewards'].max()/(1-args.discount))
        Vmin = min(0.0, dataset['rewards'].min()/(1-args.discount), Vmax-1.0/(1-args.discount))

    # Setup logger
    log_path = Path(args.log_dir) / args.env_name / ('_beta' + str(args.beta) + '_bins' + str(args.n_bins))
    log = Log(log_path, vars(args))
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)

    # Assume vector observation and action
    obs_dim, act_dim = dataset['observations'].shape[1], dataset['actions'].shape[1]
    qf = TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    target_qf = copy.deepcopy(qf).requires_grad_(False)
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                            use_tanh=True, std_type='diagonal').to(DEFAULT_DEVICE)
    dataset['actions'] = np.clip(dataset['actions'], -1+EPS, 1-EPS)  # due to tanh

    if args.n_bins>0:  # Discrete action space
        n_bins = args.n_bins
        aa0 = dataset['actions']
        dataset['actions'] = c2d(dataset['actions'], min=-1, max=1, n=n_bins)
        env = DiscreteActionGymWrapper(env, min=-1, max=1, n=n_bins, dim=act_dim)
        qf = TwinQDiscrete(obs_dim,  n_bins ** act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
        target_qf = copy.deepcopy(qf).requires_grad_(False)
        # policy = SoftmaxPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
        policy = SoftmaxPolicy(obs_dim, act_dim, n_bins=n_bins, hidden_dim=1024, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)


    # if args.n_bins>0:  # Discrete action space
    #     n_bins = args.n_bins
    #     aa0 = dataset['actions']
    #     dataset['actions'] = c2d(dataset['actions'], min=-1, max=1, n=n_bins)
    #     env = DiscreteActionGymWrapper(env, min=-1, max=1, n=n_bins)
    #     qf = TwinQDiscrete(obs_dim, n_bins**act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    #     target_qf = copy.deepcopy(qf).requires_grad_(False)
    #     policy = SoftmaxPolicy(obs_dim, act_dim*n_bins,  hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)

    rl = ATAC(
        policy=policy,
        qf=qf,
        target_qf=target_qf,
        optimizer=torch.optim.Adam,
        discount=args.discount,
        buffer_batch_size=args.batch_size,
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
    dataset = bp.train(dataset, args.n_warmstart_steps, log_fun=bp_log_fun, silence=args.disable_tqdm)  # This ensures "next_observations" is in `dataset`.

    # Main Training
    for step in trange(args.n_steps, disable=args.disable_tqdm):
        train_metrics = rl.update(**sample_batch(dataset, args.batch_size))
        if step % max(int(args.eval_period/10),1) == 0  or  step==args.n_steps-1:
            print(train_metrics)
            for k, v in train_metrics.items():
                writer.add_scalar('Train/'+k, v, step)
        if step % args.eval_period == 0:
            eval_metrics = eval_agent(env=env,
                                      agent=policy,
                                      discount=args.discount,
                                      n_eval_episodes=args.n_eval_episodes,
                                      normalize_score=lambda returns: d4rl.get_normalized_score(args.env_name, returns)*100.0)
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
    parser.add_argument('--n_bins', type=int, default=-1)



    return parser

if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())
