from gym.spaces import Dict as GymDict
from ray import tune
from ray.tune import register_env
from SMAC.model.torch_mask_updet_cc import *
from SMAC.util.r2d2_tools import *
from SMAC.model.torch_vdn_qmix_iql_model import *
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as QMIX_CONFIG
from SMAC.env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC
from SMAC.util.qmix_tools import QMixTorchPolicy_Customized, execution_plan_qmix


"""
This QMiX/VDN version is based on but different from that rllib built-in qmix_policy
1. the replay buffer is now standard localreplaybuffer instead of simplereplaybuffer
2. the QMIX loss is modified to be align with pymarl
3. provide reward standardize option
4. provide model optimizer option
5. follow DQN execution plan
"""


def run_vdn_qmix_iql(args, common_config, env_config, stop, reporter):
    if args.neural_arch not in ["GRU", "UPDeT"]:
        raise NotImplementedError()

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    episode_limit = env_config["episode_limit"]

    grouping = {
        "group_1": ["agent_{}".format(i) for i in range(n_ally)],
    }
    ## obs state setting here
    obs_space = Tuple([
                          GymDict({
                              "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
                              "state": Box(-2.0, 2.0, shape=(state_shape,)),
                              "action_mask": Box(0.0, 1.0, shape=(n_actions,))
                          })] * n_ally
                      )
    act_space = Tuple([
                          Discrete(n_actions)
                      ] * n_ally)

    # QMIX/VDN need grouping
    register_env(
        "grouped_smac",
        lambda config: SMAC(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    mixer_dict = {
        "QMIX": "qmix",
        "VDN": "vdn",
        "IQL": None
    }

    config = {
        "seed": common_config["seed"],
        "env": "grouped_smac",
        "env_config": {
            "map_name": args.map,
        },
        "model": {
            "max_seq_len": episode_limit,  # dynamic
            "custom_model_config": {
                "neural_arch": args.neural_arch,
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape
            },
        },
        "mixer": mixer_dict[args.run],
        "callbacks": common_config["callbacks"],
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": args.num_gpus,
        # "_disable_preprocessor_api": True
    }

    learning_starts = episode_limit * 32
    train_batch_size = 32 // args.batchsize_reduce
    QMIX_CONFIG.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": 5000 * episode_limit // 2,  # in timesteps
            "train_batch_size": train_batch_size,  # in sequence
            "target_network_update_freq": episode_limit * 100,  # in timesteps
            "learning_starts": learning_starts,
            "lr": 0.0005,  # default
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 50000,  # Timesteps over which to anneal epsilon.
            },
            "evaluation_interval": args.evaluation_interval,
        })

    QMIX_CONFIG["reward_standardize"] = False  # this may affect the final performance if you turn it on
    QMIX_CONFIG["training_intensity"] = None
    QMIX_CONFIG["optimizer"] = "epymarl"  # pyamrl for RMSProp or epymarl for Adam

    QMixTrainer_ = QMixTrainer.with_updates(
        name="QMIX",
        default_config=QMIX_CONFIG,
        default_policy=QMixTorchPolicy_Customized,
        get_policy_class=None,
        execution_plan=execution_plan_qmix)

    results = tune.run(QMixTrainer_, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config, verbose=1, progress_reporter=reporter)

    return results
