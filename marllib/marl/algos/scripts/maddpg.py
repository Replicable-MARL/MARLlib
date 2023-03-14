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

from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.models import ModelCatalog
from marllib.marl.algos.core.CC.maddpg import MADDPGTrainer
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.utils.log_dir_util import available_local_dir
import json
from typing import Any, Dict
from ray.tune.analysis import ExperimentAnalysis


def run_maddpg(model: Any, exp: Dict, run: Dict, env: Dict,
             stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm using Ray RLlib.
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

    ModelCatalog.register_custom_model(
        "DDPG_Model", model)

    _param = AlgVar(exp)

    episode_limit = env["episode_limit"]
    train_batch_size = _param["batch_episode"]
    learning_starts = _param["learning_starts_episode"] * episode_limit
    buffer_size = _param["buffer_size_episode"] * episode_limit
    twin_q = _param["twin_q"]
    prioritized_replay = _param["prioritized_replay"]
    smooth_target_policy = _param["smooth_target_policy"]
    n_step = _param["n_step"]
    critic_lr = _param["critic_lr"]
    actor_lr = _param["actor_lr"]
    target_network_update_freq = _param["target_network_update_freq_episode"] * episode_limit
    tau = _param["tau"]
    batch_mode = _param["batch_mode"]

    config = {
        "batch_mode": batch_mode,
        "buffer_size": buffer_size,
        "train_batch_size": train_batch_size,
        "critic_lr": critic_lr if restore is None else 1e-10,
        "actor_lr": actor_lr if restore is None else 1e-10,
        "twin_q": twin_q,
        "prioritized_replay": prioritized_replay,
        "smooth_target_policy": smooth_target_policy,
        "tau": tau,
        "target_network_update_freq": target_network_update_freq,
        "learning_starts": learning_starts,
        "n_step": n_step,
        "model": {
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(exp, env),
        },
        "zero_init_states": True,
    }
    config.update(run)

    algorithm = exp["algorithm"]
    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    if restore is not None:
        with open(restore["params_path"], 'r') as JSON:
            raw_config = json.load(JSON)
            raw_config = raw_config["model"]["custom_model_config"]['model_arch_args']
            check_config = config["model"]["custom_model_config"]['model_arch_args']
            if check_config != raw_config:
                raise ValueError("is not using the params required by the checkpoint model")
        model_path = restore["model_path"]
    else:
        model_path = None

    results = tune.run(MADDPGTrainer,
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
