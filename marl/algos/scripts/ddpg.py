from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.utils.setup_utils import AlgVar
from marl.algos.core.IL.ddpg import DDPGRNNTrainer as IDDPGTrainer


def run_ddpg(config_dict, common_config, env_dict, stop):
    _param = AlgVar(config_dict)

    train_batch_size = _param["batch_episode"]
    buffer_size = _param["buffer_size"]
    episode_limit = env_dict["episode_limit"]
    algorithm = config_dict["algorithm"]
    batch_mode = _param["batch_mode"]
    lr = _param["lr"]

    learning_starts = episode_limit * train_batch_size

    config = {
        "batch_mode": batch_mode,
        "buffer_size": buffer_size,
        "train_batch_size": train_batch_size,
        "critic_lr": lr,
        "actor_lr": lr,
        "model": {
            "max_seq_len": episode_limit,
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
        "prioritized_replay": True,
        "zero_init_states": True,
        "learning_starts": learning_starts
    }
    config.update(common_config)

    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(
        IDDPGTrainer,
        name=RUNNING_NAME,
        stop=stop,
        config=config,
        verbose=1,
        progress_reporter=CLIReporter()
    )

    return results
