from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.core.CC.mappo import MAPPOTrainer
from marl.algos.utils.log_dir_util import available_local_dir
from marl.algos.utils.setup_utils import _algos_var


def run_mappo(config_dict, common_config, env_dict, stop):
    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    _param = _algos_var(config_dict)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = config_dict["algo_args"]["batch_episode"] * env_dict["episode_limit"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    algorithm = config_dict["algorithm"]
    batch_mode = config_dict["algo_args"]["batch_mode"]
    # lr = config_dict["algo_args"]["lr"]
    iteration = config_dict["algo_args"]["iteration"]
    clip_param = config_dict["algo_args"]["clip_param"]
    vf_clip_param = config_dict["algo_args"]["vf_clip_param"]
    entropy_coeff = config_dict["algo_args"]["entropy_coeff"]
    horizon = config_dict['algo_args']['horizon']
    grad_clip = config_dict['algo_args']['grad_clip']
    use_critic = config_dict['algo_args']['use_critic']
    gamma = config_dict['algo_args']['gamma']

    config = {
        "seed": tune.grid_search([1]),
        # "seed": 123,
        # "batch_mode": batch_mode,
        "horizon": horizon,
        "num_sgd_iter": iteration,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        # "lr": _param('lr'),
        # "entropy_coeff": entropy_coeff,
        "grad_clip": grad_clip,
        "use_critic": use_critic,
        "clip_param": clip_param,
        "gamma": gamma,
        # "vf_clip_param": vf_clip_param,  # very sensitive, depends on the scale of the rewards
        "model": {
            "custom_model": "Centralized_Critic_Model",
            # "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
            "vf_share_layers": True,
        },
    }
    config.update(common_config)

    results = tune.run(MAPPOTrainer, name=algorithm + "_" + config_dict["model_arch_args"]["core_arch"] + "_" +
                                          config_dict["env_args"][
                                              "map_name"],
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir)

    return results
