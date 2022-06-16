from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.core.CC.mappo import MAPPOTrainer
from marl.algos.utils.log_dir_util import available_local_dir
from marl.algos.utils.setup_utils import AlgVar


def run_mappo(config_dict, common_config, env_dict, stop):
    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    _param = AlgVar(config_dict)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = _param["batch_episode"] * env_dict["episode_limit"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    algorithm = config_dict["algorithm"]
    batch_mode = _param["batch_mode"]
    # lr = _param["lr"]
    iteration = _param["iteration"]
    clip_param = _param["clip_param"]
    vf_clip_param = _param["vf_clip_param"]
    entropy_coeff = _param["entropy_coeff"]
    horizon = config_dict['algo_args']['horizon']
    grad_clip = config_dict['algo_args']['grad_clip']
    use_critic = config_dict['algo_args']['use_critic']
    gamma = config_dict['algo_args']['gamma']

    config = {
        "horizon": horizon,
        "num_sgd_iter": iteration,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": grad_clip,
        "use_critic": use_critic,
        "clip_param": clip_param,
        "gamma": gamma,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "custom_model_config": merge_dicts(config_dict, env_dict),
            "vf_share_layers": True,
        },
    }
    config.update(common_config)

    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(MAPPOTrainer,
                       name=RUNNING_NAME,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir)

    return results
