from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from marl.algos.core.CC.happo import HAPPOTrainer


def run_happo(config_dict, common_config, env_dict, stop):
    def _algos_var(key): return config_dict['algo_args'][key]

    sgd_minibatch_size = _algos_var('sgd_minibatch_size')
    episode_limit = _algos_var('horizon')

    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    config = {}
    config.update(common_config)

    map_name = config_dict["env_args"]["map_name"],
    arch = config_dict["model_arch_args"]["core_arch"]

    config['normal_value'] = _algos_var('normal_value')

    config.update({
        "seed": 1,
        "env": map_name,
        "horizon": episode_limit,
        "num_sgd_iter": _algos_var('num_sgd_iter'),
        "train_batch_size": _algos_var('train_batch_size'),
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": _algos_var('grad_clip'),
        "clip_param": _algos_var('clip_param'),
        "use_critic": _algos_var('use_critic'),
        "gamma": _algos_var('gamma'),
        "model": {
            "custom_model": "{}_CentralizedCritic".format(arch),
            "custom_model_config": merge_dicts(config_dict, env_dict),
            "vf_share_layers": _algos_var('vf_share_layers')
        },
    })

    PPO_CONFIG.update({
        'critic_lr': _algos_var('critic_lr'),
        'lr': _algos_var('lr'),
        "lr_schedule": [
            (_algos_var('lr_sh_min'), _algos_var('lr_sh_max')),
            (int(_algos_var('lr_sh_step')), _algos_var('lr_min')),
        ]
    })

    algorithm = config_dict["algorithm"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(HAPPOTrainer(PPO_CONFIG),
                       name=RUNNING_NAME,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter()
    )

    return results
