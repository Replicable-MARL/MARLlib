from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.core.CC.coma import COMATrainer
from marl.algos.utils.setup_utils import AlgVar


def run_coma(config_dict, common_config, env_dict, stop):
    _param = AlgVar(config_dict)

    algorithm = config_dict["algorithm"]
    episode_limit = env_dict["episode_limit"]

    train_batch_episode = config_dict["algo_args"]["batch_episode"]
    train_batch_episode = _param["batch_episode"]

    batch_mode = _param["batch_mode"]
    lr = _param["lr"]
    entropy_coeff = _param["entropy_coeff"]

    config = {
        "batch_mode": batch_mode,
        "train_batch_size": train_batch_episode * episode_limit,
        "lr": lr,
        "entropy_coeff": entropy_coeff,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }
    config.update(common_config)
    algorithm = config_dict["algorithm"]

    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(
        COMATrainer,
        name=RUNNING_NAME,
        stop=stop,
        config=config,
        verbose=1,
        progress_reporter=CLIReporter()
    )

    return results
