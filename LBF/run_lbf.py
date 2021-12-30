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
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG

from LBF.config_lbf import get_train_parser
from LBF.env.lbf_rllib import RllibLBF
from LBF.env.lbf_rllib_qmix import RllibLBF_QMIX

from LBF.model.torch_gru import *
from LBF.model.torch_gru_cc import *
from LBF.model.torch_lstm import *
from LBF.model.torch_lstm_cc import *
from LBF.util.mappo_tools import *
from LBF.util.maa2c_tools import *

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=True)

    agent_num = args.agent_num

    env_config = {
        "num_agents": args.agent_num,
        "field_size": args.field_size,
        "max_food": args.max_food,
        "sight": args.sight,
        "force_coop": args.force_coop,
    }

    map_name = "Foraging-{4}s-{0}x{0}-{1}p-{2}f{3}".format(
        args.field_size,
        args.agent_num,
        args.max_food,
        "-coop" if args.force_coop else "",
        args.sight
    )

    register_env("lbf", lambda _: RllibLBF(env_config))

    # Independent
    ModelCatalog.register_custom_model(
        "GRU_IndependentCritic", Torch_GRU_Model)
    ModelCatalog.register_custom_model(
        "LSTM_IndependentCritic", Torch_LSTM_Model)

    # CTDE(centralized critic)
    ModelCatalog.register_custom_model(
        "GRU_CentralizedCritic", Torch_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "LSTM_CentralizedCritic", Torch_LSTM_CentralizedCritic_Model)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    if args.run in ["QMIX", "VDN"]:  # policy and model are implemented as source code is

        if args.neural_arch not in ["GRU"]:
            print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
            raise ValueError()

        if not args.force_coop:
            print("competitive settings are not suitable for QMIX/VDN")
            raise ValueError()

        single_env = RllibLBF_QMIX(env_config)
        obs_space = single_env.observation_space
        act_space = single_env.action_space

        obs_space = Tuple([obs_space] * agent_num)
        act_space = Tuple([act_space] * agent_num)

        # align with LBF/env/lbf_rllib_qmix.py reset() function in line 41-50
        grouping = {
            "group_1": ["agent_{}".format(i) for i in range(agent_num)],
        }

        # QMIX/VDN algo needs grouping env
        register_env(
            "grouped_lbf",
            lambda _: RllibLBF_QMIX(env_config).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

        config = {
            "env": "grouped_lbf",
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

        results = tune.run("QMIX",
                           name=args.run + "_" + args.neural_arch + "_" + map_name,
                           stop=stop,
                           config=config,
                           verbose=1)

    else:  # "PG", "A2C", "A3C", "R2D2", "PPO"

        single_env = RllibLBF(env_config)
        obs_space = single_env.observation_space
        act_space = single_env.action_space

        policies = {
            "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(agent_num)
        }
        policy_ids = list(policies.keys())

        common_config = {
            "env": "lbf",
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

            config = {}

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": "{}_IndependentCritic".format(args.neural_arch),
                    },
                })

            config.update(common_config)
            results = tune.run(args.run,
                               name=args.run + "_" + args.neural_arch + "_" + map_name,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run in ["PPO"]:

            config = {"num_sgd_iter": 5, }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": "{}_IndependentCritic".format(args.neural_arch),
                    },
                })

            config.update(common_config)
            results = tune.run(args.run,
                               name=args.run + "_" + args.neural_arch + "_" + map_name,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run == "MAA2C":  # centralized A2C

            config = {
                "model": {
                    "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                    "custom_model_config": {
                        "agent_num": agent_num
                    },
                },
            }
            config.update(common_config)

            MAA2CTFPolicy = A3CTFPolicy.with_updates(
                name="MAA2CTFPolicy",
                postprocess_fn=centralized_critic_postprocessing,
                loss_fn=loss_with_central_critic_a2c,
                grad_stats_fn=central_vf_stats_a2c,
                mixins=[
                    CentralizedValueMixin
                ])

            MAA2CTorchPolicy = A3CTorchPolicy.with_updates(
                name="MAA2CTorchPolicy",
                get_default_config=lambda: A3C_CONFIG,
                postprocess_fn=centralized_critic_postprocessing,
                loss_fn=loss_with_central_critic_a2c,
                mixins=[
                    CentralizedValueMixin
                ])


            def get_policy_class(config_):
                if config_["framework"] == "torch":
                    return MAA2CTorchPolicy


            MAA2CTrainer = A2CTrainer.with_updates(
                name="MAA2CTrainer",
                default_policy=MAA2CTFPolicy,
                get_policy_class=get_policy_class,
            )

            results = tune.run(MAA2CTrainer,
                               name=args.run + "_" + args.neural_arch + "_" + map_name,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run in ["MAPPO"]:
            config = {
                "model": {
                    "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                    "custom_model_config": {
                        "agent_num": agent_num
                    }
                },
                "num_sgd_iter": 10,
            }
            config.update(common_config)

            MAPPOTFPolicy = PPOTFPolicy.with_updates(
                name="MAPPOTFPolicy",
                postprocess_fn=centralized_critic_postprocessing,
                loss_fn=loss_with_central_critic,
                before_loss_init=setup_tf_mixins,
                grad_stats_fn=central_vf_stats_ppo,
                mixins=[
                    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
                    CentralizedValueMixin
                ])

            MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
                name="MAPPOTorchPolicy",
                get_default_config=lambda: PPO_CONFIG,
                postprocess_fn=centralized_critic_postprocessing,
                loss_fn=loss_with_central_critic,
                before_init=setup_torch_mixins,
                mixins=[
                    TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
                    CentralizedValueMixin
                ])


            def get_policy_class(config_):
                if config_["framework"] == "torch":
                    return MAPPOTorchPolicy


            MAPPOTrainer = PPOTrainer.with_updates(
                name="MAPPOTrainer",
                default_policy=MAPPOTFPolicy,
                get_policy_class=get_policy_class,
            )

            results = tune.run(MAPPOTrainer,
                               name=args.run + "_" + args.neural_arch + "_" + map_name,
                               stop=stop,
                               config=config,
                               verbose=1)
        else:
            print("{} algo not supported".format(args.run))
            raise ValueError()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
