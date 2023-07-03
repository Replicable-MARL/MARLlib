# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marllib.marl.algos.core.CC.happo import HAPPOTrainer
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from marllib.marl.algos.scripts.coma import restore_model
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.models import ModelCatalog
import json
from typing import Any, Dict
from ray.tune.analysis import ExperimentAnalysis


def run_happo(model: Any, exp: Dict, run: Dict, env: Dict,
              stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the Heterogeneous-Agent Proximal Policy Optimisation (HAPPO) algorithm using Ray RLlib.
    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """
    _param = AlgVar(exp)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """

    ModelCatalog.register_custom_model(
        "Centralized_Critic_Model", model)

    train_batch_size = _param["batch_episode"] * env["episode_limit"]
    if "fixed_batch_timesteps" in exp:
        train_batch_size = exp["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    batch_mode = _param["batch_mode"]
    lr = _param["lr"]
    clip_param = _param["clip_param"]
    grad_clip = _param["grad_clip"]
    use_gae = _param["use_gae"]
    critic_lr = _param["critic_lr"]
    gae_lambda = _param["lambda"]
    num_sgd_iter = _param["num_sgd_iter"]
    vf_loss_coeff = _param["vf_loss_coeff"]
    entropy_coeff = _param["entropy_coeff"]
    vf_clip_param = _param["vf_clip_param"]
    gamma = _param["gamma"]

    exp['actor_lr'] = lr
    exp['critic_lr'] = critic_lr
    exp['gain'] = _param['gain']

    seed = random.randint(0, 10)
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    config = {
        "seed": seed,
        "horizon": episode_limit,
        "batch_mode": batch_mode,
        "use_gae": use_gae,
        "lambda": gae_lambda,
        "gamma": gamma,
        "vf_loss_coeff": vf_loss_coeff,
        "vf_clip_param": vf_clip_param,
        "entropy_coeff": entropy_coeff,
        "lr": critic_lr if restore is None else 1e-10,
        "num_sgd_iter": num_sgd_iter,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": grad_clip,
        "clip_param": clip_param,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": back_up_config,
            "vf_share_layers": True,
        },
    }
    config.update(run)

    TRAIN_MARK = 'append-data'

    PPO_CONFIG.update({
        'lr': critic_lr,
        "lr_schedule": [
            (0, critic_lr),
            (int(exp['stop_timesteps']), _param['min_lr_schedule']),
        ]
    })

    algorithm = exp["algorithm"]
    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name, str(lr), str(critic_lr), TRAIN_MARK.upper(), f'seed-{seed}'])

    model_path = restore_model(restore, exp)

    _HAPPOTrainer = HAPPOTrainer(PPO_CONFIG)

    results = tune.run(_HAPPOTrainer,
                       name=RUNNING_NAME,
                       checkpoint_at_end=exp['checkpoint_end'],
                       checkpoint_freq=exp['checkpoint_freq'],
                       restore=model_path,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"])

    return results
