"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.catalog import ModelCatalog
from gym.spaces import Dict as GymDict, Tuple, Box, Discrete
from ray.rllib.utils.test_utils import check_learning_achieved
import ray
from ray import tune
from ray.tune.registry import register_env
from GoogleFootball.config_football import get_train_parser
from GoogleFootball.env.football_rllib import RllibGFootball
from GoogleFootball.env.football_rllib_qmix import RllibGFootball_QMIX

from GoogleFootball.model.torch_qmix_model import QMixTrainer
from GoogleFootball.model.torch_cnn_lstm import Torch_CNN_LSTM_Model
from GoogleFootball.model.torch_cnn_gru import Torch_CNN_GRU_Model
from GoogleFootball.model.torch_cnn_updet import Torch_CNN_Transformer_Model

ally_num_dict = {
    "academy_empty_goal": 2,
    "academy_empty_goal_close": 2,
    "academy_run_to_score_with_keeper": 2,
    "academy_run_to_score": 2,
    "academy_pass_and_shoot_with_keeper": 3,
    "academy_run_pass_and_shoot_with_keeper": 3,
    "academy_3_vs_1_with_keeper": 4,
    "academy_single_goal_versus_lazy": 11,
    "academy_corner": 11,
    "academy_counterattack_easy": 11,
    "academy_counterattack_hard": 11,
}

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=True)

    ally_num = ally_num_dict[args.map]

    env_config = {
        "env_name": args.map,
        "num_agents": ally_num
    }

    register_env("football", lambda _: RllibGFootball(env_config))

    # Independent
    ModelCatalog.register_custom_model("CNN_LSTM", Torch_CNN_LSTM_Model)
    ModelCatalog.register_custom_model("CNN_GRU", Torch_CNN_GRU_Model)
    ModelCatalog.register_custom_model("CNN_UPDeT", Torch_CNN_Transformer_Model)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    if args.run in ["QMIX", "VDN"]:  # policy and model are implemented as source code is

        if args.neural_arch not in ["CNN_GRU", "CNN_UPDeT"]:
            print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
            raise ValueError()

        single_env = RllibGFootball_QMIX(env_config)
        obs_space = single_env.observation_space
        act_space = single_env.action_space

        obs_space = Tuple([obs_space] * ally_num)
        act_space = Tuple([act_space] * ally_num)

        # align with GoogleFootball/env/football_rllib_qmix.py reset() function in line 41-50
        grouping = {
            "group_1": ["agent_{}".format(i) for i in range(ally_num)],
        }

        # QMIX/VDN algo needs grouping env
        register_env(
            "grouped_football",
            lambda _: RllibGFootball_QMIX(env_config).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

        config = {
            "env": "grouped_football",
            "train_batch_size": 32,
            "exploration_config": {
                "epsilon_timesteps": 5000,
                "final_epsilon": 0.05,
            },
            "model": {
                "custom_model_config": {
                    "neural_arch": args.neural_arch,
                },
            },
            "mixer": "qmix" if args.run == "QMIX" else None,  # None for VDN, which has no mixer
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,
            "num_gpus_per_worker": args.num_gpus_per_worker,

        }

        results = tune.run(QMixTrainer,
                           name=args.run + "_" + args.neural_arch + "_" + args.map,
                           stop=stop,
                           config=config,
                           verbose=1)

    else:  # "PG", "A2C", "A3C", "R2D2", "PPO"

        single_env = RllibGFootball(env_config)
        obs_space = single_env.observation_space
        act_space = single_env.action_space

        policies = {
            "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(ally_num)
        }
        policy_ids = list(policies.keys())

        common_config = {
            "num_gpus_per_worker": args.num_gpus_per_worker,
            "train_batch_size": 1000,
            "num_workers": args.num_workers,
            "num_gpus": args.num_gpus,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": tune.function(
                    lambda agent_id: policy_ids[int(agent_id[6:])]),
            },
            "framework": args.framework,
        }

        if args.run in ["PG", "A2C", "A3C", "R2D2"]:

            config = {
                "env": "football",
            }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": args.neural_arch,
                    },
                })

            config.update(common_config)
            results = tune.run(args.run,
                               name=args.run + "_" + args.neural_arch + "_" + args.map,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run in ["PPO"]:

            config = {
                "env": "football",
                "num_sgd_iter": 5,
            }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": args.neural_arch,
                    },
                })

            config.update(common_config)
            results = tune.run(args.run,
                               name=args.run + "_" + args.neural_arch + "_" + args.map,
                               stop=stop,
                               config=config,
                               verbose=1)

        else:
            print("{} algo not supported".format(args.run))
            raise ValueError()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
