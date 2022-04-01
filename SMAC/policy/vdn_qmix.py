from gym.spaces import Dict as GymDict
from ray import tune
from ray.tune import register_env
from SMAC.model.torch_mask_updet_cc import *
from SMAC.util.r2d2_tools import *
from SMAC.model.torch_qmix_mask_gru_updet import *
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as QMIX_CONFIG

from SMAC.env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC
import os


def run_vdn_qmix(args, common_config, env_config, stop):
    if args.neural_arch not in ["GRU", "UPDeT"]:
        assert NotImplementedError

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

    config = {
        "seed": common_config["seed"],
        "env": "grouped_smac",
        "env_config": {
            "map_name": args.map,
        },
        "model": {
            "custom_model_config": {
                "neural_arch": args.neural_arch,
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape
            },
        },
        "mixer": "qmix" if args.run == "QMIX" else None,  # VDN has no mixer network
        "callbacks": common_config["callbacks"],
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": args.num_gpus,
    }

    QMIX_CONFIG.update(
        {
            "buffer_size": 5000,
            "train_batch_size": episode_limit * 32,  # in timesteps
            "target_network_update_freq": episode_limit * 200,  # in timesteps
            "exploration_config": {
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 50000,  # Timesteps over which to anneal epsilon.
            },
            "evaluation_interval": args.evaluation_interval,
        })

    QMixTrainer_ = QMixTrainer.with_updates(
        name="QMIX",
        default_config=QMIX_CONFIG,
        default_policy=Customized_QMixTorchPolicy,
        get_policy_class=None,
        execution_plan=execution_plan)

    results = tune.run(QMixTrainer_, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config, verbose=1)

    return results
