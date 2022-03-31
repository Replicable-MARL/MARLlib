"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.catalog import ModelCatalog
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

from DerksGym.config_derk import get_train_parser
from DerksGym.env.derk_rllib import RllibDerk

from DerksGym.model.torch_gru import *
from DerksGym.model.torch_gru_cc import *
from DerksGym.model.torch_lstm import *
from DerksGym.model.torch_lstm_cc import *
from DerksGym.util.mappo_tools import *
from DerksGym.util.maa2c_tools import *
import sys

if __name__ == "__main__":
    args = get_train_parser().parse_args()

    if args.num_workers > 1:
        print("Derk is based on chromium, multiple env instance is illegal")
        sys.exit()
    
    ray.init(local_mode=args.local_mode)

    env_config = {
        "agents_num": 6,
    }

    register_env("derk", lambda _: RllibDerk(env_config))

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

    if args.run in ["QMIX", "VDN", "R2D2"]:  # policy and model are implemented as source code is

        print("Continues action space exists Derk's, which is illegal in {}".format(args.run))
        sys.exit()

    else:  # "PG", "A2C", "PPO"

        common_config = {
            "env": "derk",
            "num_gpus_per_worker": args.num_gpus_per_worker,
            "train_batch_size": 1000,
            "num_workers": args.num_workers,
            "num_gpus": args.num_gpus,
            "multiagent": {
                "policies": {"shared_policy"},
                "policy_mapping_fn": (
                    lambda agent_id, episode, **kwargs: "shared_policy"),
            },
            "framework": args.framework,
        }

        if args.run in ["PG", "A2C", "R2D2"]:

            config = {}

            config.update({
                "model": {
                    "custom_model": "{}_IndependentCritic".format(args.neural_arch),
                },
            })

            config.update(common_config)
            results = tune.run(args.run,
                               name=args.run + "_" + args.neural_arch + "_" + "Derk",
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
                               name=args.run + "_" + args.neural_arch + "_" + "Derk",
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
                               name=args.run + "_" + args.neural_arch + "_" + "Derk",
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
                "num_sgd_iter": 5,
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
                               name=args.run + "_" + args.neural_arch + "_" + "Derk",
                               stop=stop,
                               config=config,
                               verbose=1)
        else:
            print("{} algo not supported".format(args.run))
            sys.exit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
