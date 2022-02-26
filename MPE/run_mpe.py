from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
import os
import sys
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG

from ray.rllib.agents.dqn.r2d2 import DEFAULT_CONFIG, R2D2Trainer
from ray.rllib.agents.dqn.r2d2_torch_policy import R2D2TorchPolicy
from ray.rllib.agents.dqn.r2d2_tf_policy import R2D2TFPolicy

from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_v2, simple_push_v2, simple_tag_v2, \
    simple_spread_v2, simple_reference_v2, simple_world_comm_v2, simple_speaker_listener_v3
import supersuit as ss

from config_mpe import *
from MPE.model.torch_gru import *
from MPE.model.torch_gru_cc import *
from MPE.model.torch_lstm import *
from MPE.model.torch_lstm_cc import *
from MPE.utils.mappo_tools import *
from MPE.utils.maa2c_tools import *

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def run(args):
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

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

    if args.map == "simple_adversary":
        env = simple_adversary_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_crypto":
        env = simple_crypto_v2.env(continuous_actions=args.continues)
    elif args.map == "simple":
        env = simple_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_push":
        env = simple_push_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_tag":
        env = simple_tag_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_spread":
        env = simple_spread_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_reference":
        env = simple_reference_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_world_comm":
        env = simple_world_comm_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_speaker_listener":
        env = simple_speaker_listener_v3.env(continuous_actions=args.continues)

    else:
        assert NotImplementedError
        print("Scenario {} not exists in pettingzoo".format(args.map))
        sys.exit()

    # keep obs and action dim same across agents
    # pad_action_space_v0 will auto mask the padding actions
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)

    register_env(args.map,
                 lambda _: PettingZooEnv(env))

    test_env = PettingZooEnv(env)
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    n_agents = len(test_env.agents)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    common_config = {
        "env": args.map,
        "num_gpus_per_worker": 0.2,
        "num_gpus": 0.6,
        "num_workers": 0,
        "train_batch_size": 1000,
        "rollout_fragment_length": 30,
        "horizon": 200,
        "multiagent": {
            "policies": {
                agent_name: (None, obs_space, act_space, {}) for agent_name in test_env.agents
            },
            "policy_mapping_fn": lambda agent_id: agent_id
        },
        # "callbacks": SmacCallbacks,
        "framework": args.framework,
    }

    if args.run in ["QMIX", "VDN"]:
        if args.map not in ["simple_spread", "simple_speaker_listener", "simple_reference"]:
            print(
                "adversarial agents contained in this MPE scenario. "
                "Not suitable for cooperative only algo {}".format(args.run)
            )
            sys.exit()
        else:
            print(
                "PettingZooEnv step function only return one agent info, "
                "not currently good for joint Q learning algo like QMIX/VDN"
                "and not compatible with rllib built-in algo"
                "\nwe are working on wrapping the PettingZooEnv"
                "to support some cooperative scenario based on Ray"
            )
            sys.exit()

    elif args.run in ["R2D2"]:  # similar to IQL in recurrent/POMDP mode

        if args.continues:
            print(
                "{} do not support continue action space".format(args.run)
            )
            sys.exit()

        config = {
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
            "framework": args.framework,
        }
        config.update(common_config)

        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return R2D2TorchPolicy

        # DEFAULT_CONFIG['dueling'] = False
        R2D2Trainer_ = R2D2Trainer.with_updates(
            name="R2D2_Trainer",
            default_config=DEFAULT_CONFIG,
            default_policy=R2D2TFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(R2D2Trainer_, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config,
                           verbose=1)


    elif args.run in ["PG", "A2C", "A3C"]:  # PG need define action mask GRU / only torch now

        config = {
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
        }

        config.update(common_config)

        tune.run(
            args.run,
            name=args.run + "_" + args.neural_arch + "_" + args.map,
            stop=stop,
            config=config,
            verbose=1
        )

    elif args.run == "DDPG":

        if not args.continues:
            print(
                "{} only support continues action space".format(args.run)
            )
            sys.exit()

        tune.run(
            args.run,
            name=args.run + "_" + args.neural_arch + "_" + args.map,
            stop=stop,
            config=common_config,
            verbose=1
        )

    elif args.run == "MADDPG":

        if not args.continues:
            print(
                "{} only support continues action space".format(args.run)
            )
            sys.exit()

        from MPE.model.torch_maddpg import DDPGCentralizedCriticModel
        ModelCatalog.register_custom_model(
            "torch_maddpg", DDPGCentralizedCriticModel)

        config = {
            "model": {
                "custom_model": "torch_maddpg",
                "custom_model_config": {
                    "agent_num": n_agents
                },
            },
        }
        config.update(common_config)

        from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
        from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy, ComputeTDErrorMixin
        from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
        from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG as DDPG_CONFIG
        from MPE.utils.maddpg_tools import loss_with_central_critic_ddpg
        from ray.rllib.agents.sac.sac_torch_policy import TargetNetworkMixin

        # from ray.tune.registry import register_trainable
        # register_trainable("MADDPG", MADDPGTrainer)

        MADDPGTFPolicy = DDPGTFPolicy.with_updates(
            name="MADDPGTFPolicy",
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic_ddpg,
            mixins=[
                TargetNetworkMixin,
                ComputeTDErrorMixin,
                CentralizedValueMixin
            ])

        MADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
            name="MADDPGTorchPolicy",
            get_default_config=lambda: DDPG_CONFIG,
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic_ddpg,
            mixins=[
                TargetNetworkMixin,
                ComputeTDErrorMixin,
                CentralizedValueMixin
            ])

        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return MADDPGTorchPolicy

        MADDPGTrainer = DDPGTrainer.with_updates(
            name="MADDPGTrainer",
            default_policy=MADDPGTFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(MADDPGTrainer,
                           name=args.run + "_" + args.map,
                           stop=stop,
                           config=config,
                           verbose=1)

    elif args.run == "MAA2C":  # centralized A2C

        config = {
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "custom_model_config": {
                    "agent_num": n_agents
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

    elif args.run in ["PPO"]:
        config = {
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
            "num_sgd_iter": 10,
        }

        config.update(common_config)

        tune.run(
            args.run,
            name=args.run + "_" + args.neural_arch + "_" + args.map,
            stop=stop,
            config=config,
            verbose=1
        )

    elif args.run in ["MAPPO"]:
        config = {
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "custom_model_config": {
                    "agent_num": n_agents
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
                           name=args.run + "_" + args.neural_arch + "_" + args.map,
                           stop=stop,
                           config=config,
                           verbose=1)

    ray.shutdown()
