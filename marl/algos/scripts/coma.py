from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.core.CC.coma import COMATrainer


def run_coma(config_dict, common_config, env_dict, stop):
    algorithm = config_dict["algorithm"]
    episode_limit = env_dict["episode_limit"]
    train_batch_episode = config_dict["algo_args"]["batch_episode"]
    batch_mode = config_dict["algo_args"]["batch_mode"]
    lr = config_dict["algo_args"]["lr"]
    entropy_coeff = config_dict["algo_args"]["entropy_coeff"]

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

    results = tune.run(COMATrainer, name=algorithm + "_" + config_dict["model_arch_args"]["core_arch"] + "_" +
                                         config_dict["env_args"][
                                             "map_name"],
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter())

    return results
