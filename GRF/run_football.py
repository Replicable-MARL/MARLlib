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
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from ray.tune.utils import merge_dicts
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG

from GRF.config_football import get_train_parser
from GRF.env.football_rllib import RllibGFootball
from GRF.env.football_rllib_qmix import RllibGFootball_QMIX

from GRF.model.torch_qmix_model import QMixTrainer
from GRF.model.torch_cnn_lstm import Torch_CNN_LSTM_Model
from GRF.model.torch_cnn_lstm_cc import Torch_CNN_LSTM_CentralizedCritic_Model
from GRF.model.torch_cnn_gru import Torch_CNN_GRU_Model
from GRF.model.torch_cnn_gru_cc import Torch_CNN_GRU_CentralizedCritic_Model
from GRF.model.torch_cnn_updet import Torch_CNN_Transformer_Model
from GRF.model.torch_cnn_updet_cc import Torch_CNN_Transformer_CentralizedCritic_Model
from GRF.model.torch_vd_ppo_a2c_cnn_gru_lstm_updet import *
from GRF.util.vda2c_tools import *
from GRF.util.vdppo_tools import *
from GRF.util.mappo_tools import *
from GRF.util.maa2c_tools import *
from GRF.util.coma_tools import loss_with_central_critic_coma, central_vf_stats_coma, COMATorchPolicy

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

# TODO VDA2C VDPPO (only action)
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

    # CTDE(centralized critic (only action))
    ModelCatalog.register_custom_model(
        "CNN_GRU_CentralizedCritic", Torch_CNN_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "CNN_LSTM_CentralizedCritic", Torch_CNN_LSTM_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "CNN_UPDeT_CentralizedCritic", Torch_CNN_Transformer_CentralizedCritic_Model)
    #
    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("CNN_GRU_ValueMixer", Torch_CNN_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("CNN_LSTM_ValueMixer", Torch_CNN_LSTM_Model_w_Mixer)
    ModelCatalog.register_custom_model("CNN_UPDeT_ValueMixer", Torch_CNN_Transformer_Model_w_Mixer)

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

        elif args.run == "MAA2C":  # centralized A2C

            config = {
                "env": "football",
            }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                        "custom_model_config": {
                            "agent_num": ally_num
                        }
                    },
                })
            else:
                raise NotImplementedError

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
                               name=args.run + "_" + args.neural_arch + "_" + args.map,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run in ["SUM-VDA2C", "MIX-VDA2C"]:

            config = {
                "env": "football",
            }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": "{}_ValueMixer".format(args.neural_arch),
                        "custom_model_config": {
                            "n_agents": ally_num,
                            "mixer": "qmix" if args.run == "MIX-VDA2C" else "vdn",
                            "mixer_emb_dim": 64,
                        },
                    },
                })
            else:
                raise NotImplementedError

            config.update(common_config)

            VDA2C_CONFIG = merge_dicts(
                A2C_CONFIG,
                {
                    "agent_num": ally_num,
                }
            )

            VDA2CTFPolicy = A3CTFPolicy.with_updates(
                name="VDA2CTFPolicy",
                postprocess_fn=value_mix_centralized_critic_postprocessing,
                loss_fn=value_mix_actor_critic_loss,
                grad_stats_fn=central_vf_stats_a2c, )

            VDA2CTorchPolicy = A3CTorchPolicy.with_updates(
                name="VDA2CTorchPolicy",
                get_default_config=lambda: VDA2C_CONFIG,
                postprocess_fn=value_mix_centralized_critic_postprocessing,
                loss_fn=value_mix_actor_critic_loss,
                mixins=[ValueNetworkMixin, MixingValueMixin],
            )


            def get_policy_class(config_):
                if config_["framework"] == "torch":
                    return VDA2CTorchPolicy


            VDA2CTrainer = A2CTrainer.with_updates(
                name="VDA2CTrainer",
                default_policy=VDA2CTFPolicy,
                get_policy_class=get_policy_class,
            )

            results = tune.run(VDA2CTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                               config=config, verbose=1)

        elif args.run in ["SUM-VDPPO", "MIX-VDPPO"]:

            """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """

            config = {
                "env": "football",
                "num_sgd_iter": 5,
            }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": "{}_ValueMixer".format(args.neural_arch),
                        "custom_model_config": {
                            "n_agents": ally_num,
                            "mixer": "qmix" if args.run == "MIX-VDA2C" else "vdn",
                            "mixer_emb_dim": 64,
                        },
                    },
                })
            else:
                raise NotImplementedError

            config.update(common_config)

            VDPPO_CONFIG = merge_dicts(
                PPO_CONFIG,
                {
                    "agent_num": ally_num,
                }
            )

            # not used
            VDPPOTFPolicy = PPOTFPolicy.with_updates(
                name="VDPPOTFPolicy",
                postprocess_fn=value_mix_centralized_critic_postprocessing,
                loss_fn=value_mix_ppo_surrogate_loss,
                before_loss_init=setup_tf_mixins,
                grad_stats_fn=central_vf_stats_ppo,
                mixins=[
                    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
                    ValueNetworkMixin, MixingValueMixin
                ])

            VDPPOTorchPolicy = PPOTorchPolicy.with_updates(
                name="VDPPOTorchPolicy",
                get_default_config=lambda: VDPPO_CONFIG,
                postprocess_fn=value_mix_centralized_critic_postprocessing,
                loss_fn=value_mix_ppo_surrogate_loss,
                before_init=setup_torch_mixins,
                mixins=[
                    TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
                    ValueNetworkMixin, MixingValueMixin
                ])


            def get_policy_class(config_):
                if config_["framework"] == "torch":
                    return VDPPOTorchPolicy


            VDPPOTrainer = PPOTrainer.with_updates(
                name="VDPPOTrainer",
                default_policy=VDPPOTFPolicy,
                get_policy_class=get_policy_class,
            )

            results = tune.run(VDPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
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

        elif args.run in ["MAPPO"]:

            config = {
                "env": "football",
                "num_sgd_iter": 5,
            }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                        "custom_model_config": {
                            "agent_num": ally_num
                        }
                    },
                })
            else:
                raise NotImplementedError

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
                               name=args.run + "_" + args.neural_arch + "_" + args.map,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run == "COMA":

            config = {
                "env": "football",
            }

            if "_" in args.neural_arch:
                config.update({
                    "model": {
                        "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                        "custom_model_config": {
                            "agent_num": ally_num,
                            "coma": True
                        },
                    },
                })
            else:
                raise NotImplementedError

            config.update(common_config)

            # not used
            COMATFPolicy = A3CTFPolicy.with_updates(
                name="MAA2CTFPolicy",
                postprocess_fn=centralized_critic_postprocessing,
                loss_fn=loss_with_central_critic_coma,
                grad_stats_fn=central_vf_stats_coma,
                mixins=[
                    CentralizedValueMixin
                ])

            COMATorchPolicy = COMATorchPolicy.with_updates(
                name="MAA2CTorchPolicy",
                loss_fn=loss_with_central_critic_coma,
                mixins=[
                    CentralizedValueMixin
                ])


            def get_policy_class(config_):
                if config_["framework"] == "torch":
                    return COMATorchPolicy


            COMATrainer = A2CTrainer.with_updates(
                name="COMATrainer",
                default_policy=COMATFPolicy,
                get_policy_class=get_policy_class,
            )

            tune.run(COMATrainer,
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
