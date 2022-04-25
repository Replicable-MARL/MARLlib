from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter


def run_pg_a2c_a3c(config_dict, common_config, env_dict, stop):
    train_batch_size = config_dict["algo_args"]["batch_episode"] * env_dict["episode_limit"]
    episode_limit = env_dict["episode_limit"]

    config = {
        "batch_mode": config_dict["algo_args"]["batch_mode"],
        "train_batch_size": train_batch_size,
        "lr": config_dict["algo_args"]["lr"],
        "model": {
            "custom_model": "Universal_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }

    config.update(common_config)

    results = tune.run(config_dict["algorithm"].upper(),
                       name=config_dict["algorithm"] + "_" + config_dict["core_arch"] + "_" + config_dict["env_args"][
                           "map_name"], stop=stop, config=config,
                       verbose=1, progress_reporter=CLIReporter())

    return results
