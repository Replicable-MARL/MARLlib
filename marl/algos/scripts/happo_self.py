from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from marl.algos.core.CC.happo import HAPPOTrainer
from marl.algos.utils.setup_utils import _algos_var


def run_happo(config_dict, common_config, env_dict, stop):
    _param = _algos_var(config_dict)

    # train_batch_size = config_dict["algo_args"]["batch_episode"] * env_dict["episode_limit"]
    # sgd_minibatch_size = train_batch_size
    # episode_limit = env_dict["episode_limit"]

    # while sgd_minibatch_size < episode_limit:
    #     sgd_minibatch_size *= 2

    train_batch_size = config_dict["algo_args"]["batch_episode"] * env_dict["episode_limit"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    algorithm = config_dict["algorithm"]
    batch_mode = config_dict["algo_args"]["batch_mode"]
    lr = float(config_dict["algo_args"]["lr"])
    iteration = config_dict["algo_args"]["iteration"]
    clip_param = config_dict["algo_args"]["clip_param"]
    vf_clip_param = config_dict["algo_args"]["vf_clip_param"]
    entropy_coeff = config_dict["algo_args"]["entropy_coeff"]

    config = {
        "batch_mode": batch_mode,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": lr,
        "entropy_coeff": entropy_coeff,
        "num_sgd_iter": iteration,
        "clip_param": clip_param,
        "vf_clip_param": vf_clip_param,  # very sensitive, depends on the scale of the rewards
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }
    config.update(common_config)

    # config = {}
    # config.update(common_config)

    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]

    batch_mode = config_dict["algo_args"]["batch_mode"]
    iteration = config_dict["algo_args"]["iteration"]

    # config.update({
    #     "seed": 1,
    #     # "horizon": episode_limit,
    #     "batch_mode": batch_mode,
    #     "num_sgd_iter": iteration,
    #     "train_batch_size": train_batch_size,
    #     "sgd_minibatch_size": sgd_minibatch_size,
    #     "grad_clip": _param('grad_clip'),
    #     "clip_param": _param('clip_param'),
    #     "use_critic": _param('use_critic'),
    #     "gamma": _param('gamma'),
    #     "model": {
    #         "custom_model": "Centralized_Critic_Model",
    #         "max_seq_len": episode_limit,
    #         "custom_model_config": merge_dicts(config_dict, env_dict),
    #         "vf_share_layers": _param('vf_share_layers')
    #     },
    # })

    # PPO_CONFIG.update({
    #     'critic_lr': _param('critic_lr'),
    #     'lr': _param('lr'),
    #     "lr_schedule": [
    #         (_param('lr_sh_min'), _param('lr_sh_max')),
    #         (_param('lr_sh_step'), _param('lr_min')),
    #     ]
    # })

    algorithm = config_dict["algorithm"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(HAPPOTrainer,
                       name=RUNNING_NAME,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter()
    )

    return results
