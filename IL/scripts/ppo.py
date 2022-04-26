from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter


def run_ppo(config_dict, common_config, env_dict, stop):
    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = config_dict["algo_args"]["batch_episode"] * env_dict["episode_limit"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    config = {
        "train_batch_size": train_batch_size,
        "num_sgd_iter": config_dict["algo_args"]["iteration"],
        "sgd_minibatch_size": sgd_minibatch_size,
        "batch_mode": config_dict["algo_args"]["batch_mode"],
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
