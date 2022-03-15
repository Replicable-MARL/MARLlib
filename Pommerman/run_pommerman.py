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
from Pommerman.config_pommerman import get_train_parser
from Pommerman.env.pommerman_rllib import RllibPommerman
from Pommerman.env.pommerman_rllib_qmix import RllibPommerman_QMIX

from Pommerman.model.torch_qmix_model import QMixTrainer
from Pommerman.model.torch_cnn_lstm import Torch_CNN_LSTM_Model
from Pommerman.model.torch_cnn_gru import Torch_CNN_GRU_Model
from Pommerman.model.torch_cnn_gru_cc import Torch_CNN_GRU_CentralizedCritic_Model
from Pommerman.model.torch_cnn_lstm_cc import Torch_CNN_LSTM_CentralizedCritic_Model
from Pommerman.model.torch_cnn import Torch_CNN_Model

from Pommerman.agent.simple_agent import SimpleAgent
from Pommerman.agent.trainable_place_holder_agent import PlaceHolderAgent
from Pommerman.agent.random_agent import RandomAgent

from Pommerman.util.mappo_tools import *
from Pommerman.util.maa2c_tools import *

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=True)

    agent_position = args.agent_position

    if "One" in args.map:
        agent_set = {0, 1}
    else:
        agent_set = {0, 1, 2, 3}

    neural_agent_pos = []
    for i in agent_position:
        neural_agent_pos.append(int(i))
        agent_set.remove(int(i))
    rule_agent_pos = list(agent_set)

    # for 4 agents bomber battle, neural_agent_pos/rule_agent_pos can be [a] to [a,b,c,d]
    # for 2 agents bomber battle, neural_agent_pos should only be [0,1], and rule_agent_pos should only be []
    env_config = {
        "map": args.map,
        "neural_agent_pos": neural_agent_pos,
        "rule_agent_pos": rule_agent_pos,
        "rule_agent_type": args.builtin_ai_type  # human_rule random_rule
    }

    agent_number = len(env_config["neural_agent_pos"])

    if "One" in args.map:
        agent_list = [None, None, ]
        if set(env_config["neural_agent_pos"] + env_config["rule_agent_pos"]) != {0, 1}:
            print("Wrong bomber agent position")
            raise ValueError()

    else:
        agent_list = [None, None, None, None]
        if set(env_config["neural_agent_pos"] + env_config["rule_agent_pos"]) != {0, 1, 2, 3}:
            print("Wrong bomber agent position")
            raise ValueError()

    for agent_pos in env_config["neural_agent_pos"]:
        agent_list[agent_pos] = PlaceHolderAgent()  # fake, just for initialization

    for agent_pos in env_config["rule_agent_pos"]:
        if args.builtin_ai_type == "human_rule":
            agent_list[agent_pos] = SimpleAgent()  # Built-in AI for initialization
        elif args.builtin_ai_type == "random_rule":
            agent_list[agent_pos] = RandomAgent()  # Built-in AI for initialization

    register_env("pommerman", lambda _: RllibPommerman(env_config, agent_list))

    # Independent
    ModelCatalog.register_custom_model("CNN", Torch_CNN_Model)
    ModelCatalog.register_custom_model("CNN_LSTM", Torch_CNN_LSTM_Model)
    ModelCatalog.register_custom_model("CNN_GRU", Torch_CNN_GRU_Model)

    # CTDE(centralized critic)
    ModelCatalog.register_custom_model(
        "CNN_GRU_CentralizedCritic", Torch_CNN_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "CNN_LSTM_CentralizedCritic", Torch_CNN_LSTM_CentralizedCritic_Model)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    if args.run in ["QMIX", "VDN"]:  # policy and model are implemented as source code is

        if args.neural_arch not in ["CNN_GRU", ]:
            print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
            raise ValueError()

        if "Team" not in args.map:
            print("QMIX/VDN is only for cooperative scenarios")
            raise ValueError()

        if env_config["neural_agent_pos"] == [0, 1, 2, 3]:
            # 2 vs 2
            grouping = {
                "group_1": ["agent_{}".format(i) for i in [0, 1]],
                "group_2": ["agent_{}".format(i) for i in [2, 3]],
            }

        elif env_config["neural_agent_pos"] in [[0, 1], [2, 3]]:
            grouping = {
                "group_1": ["agent_{}".format(i) for i in [0, 1]],
            }

        else:
            print("Wrong agent position setting")
            raise ValueError

        # simulate one single env
        single_env = RllibPommerman_QMIX(env_config, agent_list)
        obs_space = single_env.observation_space
        act_space = single_env.action_space

        obs_space = Tuple([obs_space] * 2)
        act_space = Tuple([act_space] * 2)

        # QMIX/VDN algo needs grouping env
        register_env(
            "grouped_pommerman",
            lambda _: RllibPommerman_QMIX(env_config, agent_list).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

        config = {
            "env": "grouped_pommerman",
            "train_batch_size": 32,
            "exploration_config": {
                "epsilon_timesteps": 5000,
                "final_epsilon": 0.05,
            },
            "model": {
                "custom_model_config": {
                    "neural_arch": args.neural_arch,
                    "map_size": 11 if "One" not in args.map else 8,
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

        single_env = RllibPommerman(env_config, agent_list)
        obs_space = single_env.observation_space
        act_space = single_env.action_space

        policies = {
            "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(agent_number)
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
                "env": "pommerman",
            }

            config.update({
                "model": {
                    "custom_model": args.neural_arch,
                    "custom_model_config": {
                        "agent_num": 4 if "One" not in args.map else 2,
                        "map_size": 11 if "One" not in args.map else 8,
                    }
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
                "env": "pommerman",
                "num_sgd_iter": 5,
            }

            config.update({
                "model": {
                    "custom_model": args.neural_arch,
                    "custom_model_config": {
                        "agent_num": 4 if "One" not in args.map else 2,
                        "map_size": 11 if "One" not in args.map else 8,
                    }
                },
            })

            config.update(common_config)
            results = tune.run(args.run,
                               name=args.run + "_" + args.neural_arch + "_" + args.map,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run == "MAA2C":  # centralized A2C

            if env_config["rule_agent_pos"] != []:
                print("Centralized critic can not be used in scenario where rule"
                      "based agent exists"
                      "\n Set rule_agent_pos to empty list []")
                raise ValueError()

            if "Team" not in args.map:
                print("Scenario \"{}\" is under fully observed setting. "
                      "MAA2C is not suitable".format(args.map))
                raise ValueError()

            config = {
                "env": "pommerman",
                "model": {
                    "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                    "custom_model_config": {
                        "agent_num": 4 if "One" not in args.map else 2,
                        "map_size": 11 if "One" not in args.map else 8,
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
                               name=args.run + "_" + args.neural_arch + "_" + args.map,
                               stop=stop,
                               config=config,
                               verbose=1)


        elif args.run in ["MAPPO"]:

            if env_config["rule_agent_pos"] != []:
                print("Centralized critic can not be used in scenario where rule"
                      "based agent exists"
                      "\n Set rule_agent_pos to empty list []")
                raise ValueError()

            if "Team" not in args.map:
                print("Scenario \"{}\" is under fully observed setting. "
                      "MAPPO is not suitable".format(args.map))
                raise ValueError()

            config = {
                "env": "pommerman",
                "model": {
                    "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                    "custom_model_config": {
                        "agent_num": 4 if "One" not in args.map else 2,
                        "map_size": 11 if "One" not in args.map else 8,
                    },
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
                               name=args.run + "_" + args.neural_arch + "_" + args.map,
                               stop=stop,
                               config=config,
                               verbose=1)

        elif args.run == "COMA":  # centralized A2C

            if env_config["rule_agent_pos"] != []:
                print("Centralized critic can not be used in scenario where rule"
                      "based agent exists"
                      "\n Set rule_agent_pos to empty list []")
                raise ValueError()

            if "Team" not in args.map:
                print("Scenario \"{}\" is under fully observed setting. "
                      "MAA2C is not suitable".format(args.map))
                raise ValueError()

            config = {
                "env": "pommerman",
                "model": {
                    "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                    "custom_model_config": {
                        "agent_num": 4 if "One" not in args.map else 2,
                        "map_size": 11 if "One" not in args.map else 8,
                        "coma": True
                    },
                },
            }
            config.update(common_config)

            from Pommerman.util.coma_tools import loss_with_central_critic_coma, central_vf_stats_coma, COMATorchPolicy

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
