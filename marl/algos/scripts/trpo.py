from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from marl.algos.core.IL.trpo import TRPOTrainer
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from marl.algos.utils.setup_utils import AlgVar
from marl.algos.utils.log_dir_util import available_local_dir

torch, nn = try_import_torch()


def run_trpo(config_dict, common_config, env_dict, stop):
    _param = AlgVar(config_dict)

    sgd_minibatch_size = _param['sgd_minibatch_size']
    episode_limit = _param['horizon']

    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    config = {}
    config.update(common_config)

    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]

    config.update({
        "horizon": episode_limit,
        "num_sgd_iter": _param['num_sgd_iter'],
        "train_batch_size": _param['train_batch_size'],
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": _param['grad_clip'],
        "clip_param": _param['clip_param'],
        "use_critic": _param['use_critic'],
        "gamma": _param['gamma'],
        "model": {
            "custom_model": "Base_Model",
            "custom_model_config": merge_dicts(config_dict, env_dict),
            "vf_share_layers": _param['vf_share_layers'],
        },
    })

    algorithm = config_dict["algorithm"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(
        TRPOTrainer,
        name=RUNNING_NAME,
        stop=stop,
        config=config,
        verbose=1,
        progress_reporter=CLIReporter(),
        local_dir=available_local_dir
    )

    return results
