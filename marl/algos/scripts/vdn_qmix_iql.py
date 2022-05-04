from ray import tune
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as JointQ_Config
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.utils.VD.iql_vdn_qmix import JointQTrainer

"""
This version is based on but different from that rllib built-in qmix_policy
1. the replay buffer is now standard localreplaybuffer instead of simplereplaybuffer
2. the loss function is modified to be align with pymarl
3. provide reward standardize option
4. provide model optimizer option
5. follow DQN execution plan
"""


def run_joint_q(config_dict, common_config, env_dict, stop):
    algorithm = config_dict["algorithm"]
    episode_limit = env_dict["episode_limit"]
    train_batch_episode = config_dict["algo_args"]["batch_episode"]
    lr = config_dict["algo_args"]["lr"]
    buffer_size = config_dict["algo_args"]["buffer_size"]
    target_network_update_frequency = config_dict["algo_args"]["target_network_update_freq"]
    final_epsilon = config_dict["algo_args"]["final_epsilon"]
    epsilon_timesteps = config_dict["algo_args"]["epsilon_timesteps"]
    reward_standardize = config_dict["algo_args"]["reward_standardize"]
    optimizer = config_dict["algo_args"]["optimizer"]

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

    results = tune.run(Trainer,
                       name=algorithm + "_" + config_dict["model_arch_args"]["core_arch"] + "_" + config_dict["env_args"][
                           "map_name"],
                       stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=CLIReporter())

    return results
