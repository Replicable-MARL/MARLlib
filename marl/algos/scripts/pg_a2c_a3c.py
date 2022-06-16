from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.utils.log_dir_util import available_local_dir
from marl.algos.utils.setup_utils import AlgVar


def run_pg_a2c_a3c(config_dict, common_config, env_dict, stop):
    _param = AlgVar(config_dict)

    train_batch_size = _param["batch_episode"] * env_dict["episode_limit"]
    episode_limit = env_dict["episode_limit"]


    config = {
        "batch_mode": _param["batch_mode"],
        "train_batch_size": train_batch_size,
        "lr": _param["lr"],
        "model": {
            "custom_model": "Base_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }

    if "entropy_coeff" in _param:
        config["entropy_coeff"] = _param["entropy_coeff"]

    config.update(common_config)

    algorithm = config_dict["algorithm"]
    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(
        algorithm.upper(),
        name=RUNNING_NAME,
        stop=stop, config=config,
        local_dir=available_local_dir,
        verbose=1,
        progress_reporter=CLIReporter()
    )

    return results
