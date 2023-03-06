import random
from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marllib.marl.algos.core.CC.happo import HAPPOTrainer
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.utils.log_dir_util import available_local_dir
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
import json


def run_happo(model_class, config_dict, common_config, env_dict, stop, restore):
    _param = AlgVar(config_dict)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = _param["batch_episode"] * env_dict["episode_limit"]
    if "fixed_batch_timesteps" in config_dict:
        train_batch_size = config_dict["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
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

    config_dict['actor_lr'] = lr
    config_dict['critic_lr'] = critic_lr
    config_dict['gain'] = _param['gain']

    seed = random.randint(0, 10)

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
            "custom_model_config": merge_dicts(config_dict, env_dict),
            "vf_share_layers": True,
        },
    }
    config.update(common_config)

    TRAIN_MARK = 'append-data'

    PPO_CONFIG.update({
        'lr': critic_lr,
        "lr_schedule": [
            (0, critic_lr),
            (int(config_dict['stop_timesteps']), _param['min_lr_schedule']),
        ]
    })

    algorithm = config_dict["algorithm"]
    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name, str(lr), str(critic_lr), TRAIN_MARK.upper(), f'seed-{seed}'])

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

    results = tune.run(HAPPOTrainer,
                       name=RUNNING_NAME,
                       checkpoint_at_end=config_dict['checkpoint_end'],
                       checkpoint_freq=config_dict['checkpoint_freq'],
                       restore=model_path,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir if config_dict["local_dir"] == "" else config_dict["local_dir"])

    return results
