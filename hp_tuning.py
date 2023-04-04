# To set up a job, the user needs to specify `method` and `hp_dict` and then
# import `submit_xt_job`. Optionally, the user can provide additional arguments
# `submit_xt_job` takes. To run the job, go to the `rl_nexus` directory and then
# execute this python file.

import csv
import datetime
import glob
import os
import pathlib
import time

import numpy as np
import yaml
from rl_nexus.hp_tuning_tools import submit_xt_job

# hp compute
max_total_runs = 2000
n_concurrent_runs_per_node = 1
compute_target = "azb-cpu"
docker_image = "mujoco"
azure_service = (
    "dilbertbatch"  #'dilbertbatch' #'rdlbatches' # 'rdlbatches' # dilbertbatch'
)
# dilbertbatch has 8000
# dilbertbatchnds another option (10000 cores)
# centralusbatch  rdlbatches southcentralusbatch westus3batch
# number of cores
# vm_size = 'Standard_F4s_v2'  # Standard_NC6s_v2

code_paths = os.path.dirname(
    __file__
)  # This file will be uploaded as rl_nexus/lightATAC/hp_tuning.py
method = "rl_nexus.lightATAC.hp_tuning.train"  # so we can call the method below.


def train(**config):
    """A wrapper to call the main function.
    config: a dict of desired hp values.
    """
    import sys
    sys.path.append(code_paths)
    import argparse
    from rl_nexus.lightATAC.main import main
    args = argparse.Namespace(**config)
    return main(args)


def run(
    hp_tuning_mode="grid", n_seeds_per_hp=3, max_n_nodes=1, vm_size="Standard_F4s_v2"
):

    # env_list = []
    # for env in ["hopper", "halfcheetah", "walker2d"]:
    #     for quality in ["random", "medium", "medium-expert", "medium-replay", "expert"]:
    #         for version in ["v2"]:
    #             env_list.append(env+'-'+quality+'-'+version)

    env_list = ['kitchen-complete-v0', 'kitchen-mixed-v0', 'kitchen-partial-v0']

    hps_dict = dict(
        beta=(4.0** np.arange(-4,5)).tolist()
        # beta=(4.0** np.arange(4,8)).tolist()
    )

    config = dict(
        env_name=None,
        log_dir="../results",
        seed='randint',
        discount=0.99,
        hidden_dim=256,
        n_hidden=3,
        n_steps=10**6,
        batch_size=256,
        fast_lr=5e-4,
        slow_lr=5e-7,
        beta=4.0,
        eval_period=50000, #5000,
        n_eval_episodes=10,
        n_warmstart_steps=100*10**3,
        clip_v=False,
    )

    xt_setup = {
        "activate": None,
        "other-cmds": [
            "cd rl_nexus/lightATAC",
            ". install.sh",  # install mujoco210
            "pip install -e . ",
            "cd ../../",  # dilbert directory
        ],
        "conda-packages": [],
        "pip-packages": [],
        "python-path": ["../"],
    }

    for env in env_list:
        time.sleep(1)
        hps_dict['env_name'] = [env]
        job_number = submit_xt_job(
            method,
            hps_dict,
            config=config,
            n_concurrent_runs_per_node=n_concurrent_runs_per_node,
            xt_setup=xt_setup,
            hp_tuning_mode=hp_tuning_mode,
            n_seeds_per_hp=n_seeds_per_hp,
            max_total_runs=max_total_runs,
            max_n_nodes=max_n_nodes,
            azure_service=azure_service,
            vm_size=vm_size,
            code_paths=code_paths,
            docker_image=docker_image,
            compute_target=compute_target,
            # remote_run=False,
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hp_tuning_mode", type=str, default="grid")
    parser.add_argument("--n_seeds_per_hp", type=int, default=1)
    parser.add_argument("--max_n_nodes", type=int, default=1)
    parser.add_argument("--vm_size", type=str, default="Standard_F4s_v2")  # Standard_NC6s_v2
    run(**vars(parser.parse_args()))