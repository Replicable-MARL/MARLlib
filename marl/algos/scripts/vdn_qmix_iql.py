from ray import tune
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as JointQ_Config
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.core.VD.iql_vdn_qmix import JointQTrainer
from marl.algos.utils.setup_utils import AlgVar

"""
This version is based on but different from that rllib built-in qmix_policy
1. the replay buffer is now standard localreplaybuffer instead of simplereplaybuffer
2. the loss function is modified to be align with pymarl
3. provide reward standardize option
4. provide model optimizer option
5. follow DQN execution plan
"""


def run_joint_q(config_dict, common_config, env_dict, stop):

    _param = AlgVar(config_dict)

    algorithm = config_dict["algorithm"]
    episode_limit = env_dict["episode_limit"]
    train_batch_episode = _param["batch_episode"]
    lr = _param["lr"]
    buffer_size = _param["buffer_size"]
    target_network_update_frequency = _param["target_network_update_freq"]
    final_epsilon = _param["final_epsilon"]
    epsilon_timesteps = _param["epsilon_timesteps"]
    reward_standardize = _param["reward_standardize"]
    optimizer = _param["optimizer"]

    mixer_dict = {
        "qmix": "qmix",
        "vdn": "vdn",
        "iql": None
    }

    config = {
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }

    config.update(common_config)

    JointQ_Config.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": buffer_size * episode_limit,  # in timesteps
            "train_batch_size": train_batch_episode,  # in sequence
            "target_network_update_freq": episode_limit * target_network_update_frequency,  # in timesteps
            "learning_starts": episode_limit * train_batch_episode,
            "lr": lr,  # default
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": epsilon_timesteps,
            },
            "mixer": mixer_dict[algorithm]
        })

    JointQ_Config["reward_standardize"] = reward_standardize  # this may affect the final performance if you turn it on
    JointQ_Config["optimizer"] = optimizer
    JointQ_Config["training_intensity"] = None

    Trainer = JointQTrainer.with_updates(
        name=algorithm.upper(),
        default_config=JointQ_Config
    )

    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join([algorithm, arch, map_name])

    results = tune.run(Trainer,
                       name=RUNNING_NAME,
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter())

    return results
