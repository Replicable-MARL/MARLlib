from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from marl.algos.core.CC.hatrpo import HATRPOTrainer
from marl.algos.utils.log_dir_util import available_local_dir
from marl.algos.utils.setup_utils import AlgVar
from marl.algos.utils.trust_regions import TrustRegionUpdator


def run_hatrpo(config_dict, common_config, env_dict, stop):
    """
        for bug mentioned https://github.com/ray-project/ray/pull/20743
        make sure sgd_minibatch_size > max_seq_len
        """
    _param = AlgVar(config_dict)

    kl_threshold = _param['kl_threshold']
    accept_ratio = _param['accept_ratio']
    critic_lr = _param['critic_lr']

    TrustRegionUpdator.kl_threshold = kl_threshold
    TrustRegionUpdator.accept_ratio = accept_ratio
    TrustRegionUpdator.critic_lr = critic_lr

    train_batch_size = _param["batch_episode"] * env_dict["episode_limit"]
    if "fixed_batch_timesteps" in config_dict:
        train_batch_size = config_dict["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    batch_mode = _param["batch_mode"]
    clip_param = _param["clip_param"]
    grad_clip = _param["grad_clip"]
    use_gae = _param["use_gae"]
    gamma = _param["gamma"]
    gae_lambda = _param["lambda"]
    kl_coeff = _param["kl_coeff"]
    num_sgd_iter = _param["num_sgd_iter"]
    vf_loss_coeff = _param["vf_loss_coeff"]
    entropy_coeff = _param["entropy_coeff"]
    vf_clip_param = _param["vf_clip_param"]

    config = {
        "batch_mode": batch_mode,
        "use_gae": use_gae,
        "lambda": gae_lambda,
        "gamma": gamma,
        "kl_coeff": kl_coeff,
        "vf_loss_coeff": vf_loss_coeff,
        "vf_clip_param": vf_clip_param,
        "entropy_coeff": entropy_coeff,
        "num_sgd_iter": num_sgd_iter,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": grad_clip,
        "clip_param": clip_param,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }
    config.update(common_config)

    algorithm = config_dict["algorithm"]
    arch = config_dict["model_arch_args"]["core_arch"]
    map_name = config_dict["env_args"]["map_name"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(HATRPOTrainer,
                       name=RUNNING_NAME,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir,
                       )

    return results
