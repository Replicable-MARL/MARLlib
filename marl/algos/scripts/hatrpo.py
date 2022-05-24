from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from marl.algos.core.CC.hatrpo import HATRPOTrainer
from marl.algos.utils.setup_utils import _algos_var
from functools import partial


def run_hatrpo(config_dict, common_config, env_dict, stop):
    _param = _algos_var(config=config_dict)

    sgd_minibatch_size = _param('sgd_minibatch_size')
    episode_limit = _param('horizon')

    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    config = {}
    config.update(common_config)

    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]

    config.update({
        "seed": 1,
        "horizon": episode_limit,
        "num_sgd_iter": _param('num_sgd_iter'),
        "train_batch_size": _param('train_batch_size'),
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": _param('grad_clip'),
        "clip_param": _param('clip_param'),
        "use_critic": _param('use_critic'),
        "gamma": _param('gamma'),
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "custom_model_config": merge_dicts(config_dict, env_dict),
            "vf_share_layers": _param('vf_share_layers')
        },
    })

    config.update({
        'critic_lr': _param('critic_lr'),
        'lr': _param('lr'),
        "lr_schedule": [
            (_param('lr_sh_min'), _param('lr_sh_max')),
            (int(_param('lr_sh_step')), _param('lr_min')),
        ]
    })

    algorithm = config_dict["algorithm"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(HATRPOTrainer,
                       name=RUNNING_NAME,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter()
    )

    return results
