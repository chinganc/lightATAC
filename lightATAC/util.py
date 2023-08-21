import copy, csv, json, random, string, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from scipy import signal
from torch.distributions import Distribution


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-6

# Methods for processing trajectories

def traj_to_tuple_data(traj_data, ignores=("metadata",)):
    """Concatenate a list of trajectory dicts to a dict of np.arrays or torch.tensors."""
    tuple_data = dict()
    for k in traj_data[0].keys():
        if not any([ig in k for ig in ignores]):
            if torch.is_tensor(traj_data[0][k]):
                tuple_data[k] = torch.cat([traj[k] for traj in traj_data])
            else:
                tuple_data[k] = np.concatenate([traj[k] for traj in traj_data])
    return tuple_data


def tuple_to_traj_data(tuple_data, ignores=("metadata",)):
    """Split a tuple_data dict of np.arrays or torch.tensors in d4rl format to list of trajectory dicts."""
    assert 'timeouts' in tuple_data and 'terminals' in tuple_data

    tuple_data["timeouts"][-1] = not tuple_data["terminals"][-1]
    ends = (tuple_data["terminals"] + tuple_data["timeouts"]) > 0
    ends[-1] = False  # don't need to split at the end

    inds = torch.arange(len(ends))[ends] + 1
    tmp_data = dict()
    for k, v in tuple_data.items():
        if not any([ig in k for ig in ignores]):
            if torch.is_tensor(v):
                secs = np.diff(np.insert(inds, (0,len(inds)),  (0,len(v)))).tolist()
                tmp_data[k] = v.split(secs)
            else:
                tmp_data[k] = np.split(v, inds)
    traj_data = [
        dict(zip(tmp_data, t)) for t in zip(*tmp_data.values())
    ]  # convert to list of dict
    return traj_data


def add_next_observations(traj):
    if "next_observations" not in traj:
        # process 'observations'
        if traj["terminals"][-1] > 0:  # duplicate the last element
            if torch.is_tensor(traj["observations"]):
                traj["observations"] = torch.cat((traj["observations"], traj["observations"][-1:]), dim=0)
            else:
                traj["observations"] = np.append(traj["observations"], traj["observations"][-1:], axis=0)
        else:  # ends because of timeout
            for k, v in traj.items():
                if k != "observations":
                    traj[k] = v[:-1]
                traj['timeouts'][-1:] = 1
        # At this point, traj['observations'] should have one more element than the others.
        traj["next_observations"] = traj["observations"][1:]
        traj["observations"] = traj["observations"][:-1]
    lens = [len(v) for k, v in traj.items()]
    assert all([lens[0] == l for l in lens[1:]])


def traj_data_to_qlearning_data(traj_data, ignores=("metadata",)):
    """Convert a list of trajectory dicts of np.arrays or torch.tensors into d4rl qlearning data format.
       This would add a new field "next_observations".
    """
    traj_data = copy.deepcopy(traj_data)
    for traj in traj_data:
        add_next_observations(traj)
    return traj_to_tuple_data(traj_data, ignores=ignores)

def cat_data_dicts(*data_dicts):
    new_data = dict()
    for k in data_dicts[0]:
        if torch.is_tensor(data_dicts[0][k]):
            new_data[k] = torch.cat([d[k] for d in data_dicts])
        else:
            new_data[k] = np.concatenate([d[k] for d in data_dicts])
    return new_data

def normalized_sum(loss, reg, w):
    return loss/w + reg if w>1 else loss + w*reg

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def discount_cumsum(x, discount):
    """Discounted cumulative sum.
    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.
    Returns:
        np.ndarrary: Discounted cumulative sum.
    """
    if torch.is_tensor(x):
        return torchaudio.functional.lfilter(
                x.flip(dims=(0,)),
                a_coeffs=torch.tensor([1, -discount], device=x.device),
                b_coeffs=torch.tensor([1, 0], device=x.device), clamp=False).flip(dims=(0,))
    else:
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=-1)[::-1]


def split(tensor_or_distribution, split_size_or_sections):
    if torch.is_tensor(tensor_or_distribution):
        return torch.split(tensor_or_distribution, split_size_or_sections)
    else:
        return split_dist(tensor_or_distribution, split_size_or_sections)


def split_dist(dist, split_size_or_sections):
    """ Split a Distribution object. """
    batch_shape = dist.batch_shape

    if isinstance(split_size_or_sections, int):
        size = split_size_or_sections
        batch_size = batch_shape[0]
        max_section = batch_size // size
        st = 0
        slices = []
        for _ in range(size):
            ed = min(st + max_section, batch_size)
            if _ == size-1:
                assert ed==batch_size
            slices.append(slice(st, ed))
            st = ed
    elif isinstance(split_size_or_sections, (list, tuple)):
        sections = split_size_or_sections
        assert sum(sections)==batch_shape[0]
        st = 0
        slices = []
        for section in sections:
            ed = st + section
            slices.append(slice(st, ed))
            st = ed
    else:
        raise ValueError("split_size_or_sections should be int, or list/tuple of int")
    return ( slice_dist(dist, s) for s in slices)


def slice_dist(dist, slice):
    """ Recursively slicing a Distribution object. """
    batch_shape = dist.batch_shape
    new = type(dist).__new__(type(dist))
    for k, v in dist.__dict__.items():
        if isinstance(v, torch.distributions.Distribution):
            sliced_v = slice_dist(v, slice)
        elif isinstance(v, torch.Tensor) and batch_shape == v.shape[:len(batch_shape)]:
            sliced_v = v[slice]
        elif 'batch_shape' in k:
            sliced_v = torch.zeros(v)[slice].shape
        else:
            sliced_v = v
        setattr(new, k, sliced_v)
    return new


# Below are modified from gwthomas/IQL-PyTorch

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, *inputs):
    """
        Args:
            f : The function to evaluate, which returns a tensor or a list of tensors.

            Suppose output = f(arg1, arg2), and we have list_arg1, list_arg2 which we wish to batch, where
            we assume len(list_arg_1) = len(list_arg_2). Then
                tuple_outputs = compute_batched(f, list_arg_1, list_arg_2)
            where len(tuple_outputs) = len(list_arg_1).

        Returns:
            A tuple of the original outputs of f.

    """
    # return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])
    if len(inputs)>1:  # assert the number of
        lens =  [len(xs) for xs in inputs]
        assert all(x == lens[0] for x in lens)

    outputs = f(*[torch.cat(xs, dim=0) for xs in inputs])
    if torch.is_tensor(outputs) or isinstance(outputs, torch.distributions.Distribution):
        return split(outputs, [len(x) for x in inputs[0]])
    else:  # suppose that's iterable.
        outputs = (split(o, [len(x) for x in inputs[0]]) for o in outputs)
        return tuple(zip(*outputs))




def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    assert type(x) is np.ndarray or torch.is_tensor(x), "Unsupported type: {}".format(type(x))
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if x.device != DEFAULT_DEVICE:
        x = x.to(device=DEFAULT_DEVICE)
    if x.dtype is torch.float64:
        x = x.float()
    return x


def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), DEFAULT_DEVICE  # dataset[k].device
    for v in dataset.values():
        assert len(v) == n, "Dataset values must have same length"
    indices = np.random.randint(low=0, high=n, size=(batch_size,))  # , device=device)
    return {k: torchify(v[indices]) for k, v in dataset.items()}


def evaluate_policy(env, policy, max_episode_steps, deterministic=True, discount = 0.99):
    obs = env.reset()
    total_reward = 0.
    discount_total_reward = 0.
    for i in range(max_episode_steps):
        with torch.no_grad():
            try:
                action = policy.act(torchify(obs), deterministic=deterministic).cpu().numpy()
            except:
                action = policy.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        discount_total_reward += reward * discount**i
        if done:
            break
        else:
            obs = next_obs
    return [total_reward, discount_total_reward]


def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'


class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()

class MixtureDistribution(Distribution):
    """
    A wrapper class for MixtureDistribution which implements
    reparameterized sampling
    """
    def __init__(self, dist, use_tanh=False, action_scale=1.):
        self.dist = dist
        self.has_rsample = True
        self.use_tanh = use_tanh
        self.action_scale = action_scale

        # XXX: assumes samples from dist is flat
        assert len(self.dist.event_shape) == 1

    def sample(self, sample_shape: torch.Size = torch.Size()):
        return self.dist.sample(sample_shape=sample_shape)

    def rsample(self):
        """
        rsample from each component distribution, and append the
        mixture probability to the end
        # XXX: does not take in SampleSize at the moment
        """
        component_rsample = self.dist.component_distribution.rsample()
        if self.use_tanh:
            # Clip samples for tanh distribution to ensure stability
            component_rsample = torch.clamp(component_rsample,
                                            min=-self.action_scale+EPS,
                                            max=self.action_scale-EPS)
        rsample = torch.cat((component_rsample,
                             self.dist.mixture_distribution.probs.unsqueeze(-1)),
                             dim=-1)
        return rsample

    def log_prob(self, value):
        """
        Compute log probability for either direct samples
        or reparameterized samples (given by self.rsample())
        """
        if value.shape[-1] == self.dist.event_shape[0]:
            # Computes log prob for samples from mixture distribution
            return self.dist.log_prob(value)
        else:
            # Computes log prob for samples for each component distribution along with mixture distribution
            # This is used when computing log prob for rsamples
            # log_prob = \sum_i w_i \sum_{x_i~P_i} log_prob(x_i)

            # Split component values and mixture probs
            component_value = value[..., :-1].transpose(0, 1) # need shape broadcastable with self.dist

            mixture_prob = value[..., -1]
            component_log_prob = self.dist.log_prob(component_value).transpose(0, 1) # transpose back
            return (component_log_prob * mixture_prob).sum(dim=-1)

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def __repr__(self):
        args_string = '\n  {},\n  {}'.format(self.dist.mixture_distribution,
                                             self.dist.component_distribution)
        return 'MixtureDistribution' + '(' + args_string + ')'


def expected_value(f, observations, gmm_rsample_actions):
    """
    Computes the expected value of f(observations, sampled_actions),
    Mainly used for computing the expected Q value for a GMM policy.
    """
    # Repeat observations to match action batch size
    # XXX: does not work if there is time dimension after batch dim or the observation is an image
    assert len(observations.shape) == 2 and len(gmm_rsample_actions.shape) == 3

    batch_size, obs_dim = observations.shape
    _, n_modes, act_dim_plus_1 = gmm_rsample_actions.shape
    observations = torch.tile(observations.unsqueeze(1),
                              [1, n_modes, 1])
    # Split actions and mixture probs
    actions_all = gmm_rsample_actions[..., :-1]
    actions_probs = gmm_rsample_actions[..., -1]

    obs_flat = observations.view(batch_size * n_modes, obs_dim)
    act_flat = actions_all.view(batch_size * n_modes, act_dim_plus_1 - 1)
    # Evaluate all actions and take expectation
    f_all = f(obs_flat, act_flat)
    if isinstance(f_all, torch.Tensor):
        expected_f = (f_all.reshape(batch_size, n_modes) * actions_probs).sum(dim=-1)
    elif isinstance(f_all, tuple):
        expected_f = tuple((ff.reshape(batch_size, n_modes) * actions_probs).sum(dim=-1) for ff in f_all)
    else:
        raise NotImplementedError
    return expected_f
