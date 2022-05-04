from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.utils.VD.vda2c import VDA2CTrainer


def run_vda2c(config_dict, common_config, env_dict, stop):
    algorithm = config_dict["algorithm"]
    episode_limit = env_dict["episode_limit"]
    train_batch_episode = config_dict["algo_args"]["batch_episode"]
    batch_mode = config_dict["algo_args"]["batch_mode"]
    lr = config_dict["algo_args"]["lr"]

    config = {
        "batch_mode": batch_mode,
        "train_batch_size": train_batch_episode * episode_limit,
        "lr": lr,
        "entropy_coeff": 0.01,
        "model": {
            "custom_model": "Onpolicy_Universal_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }
    config.update(common_config)

    results = tune.run(VDA2CTrainer, name=algorithm + "_" + config_dict["model_arch_args"]["core_arch"] + "_" +
                                          config_dict["env_args"][
                                              "map_name"],
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter())

    return results
