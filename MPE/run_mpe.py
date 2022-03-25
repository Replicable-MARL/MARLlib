from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
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
from gym.spaces import Tuple

from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_v2, simple_push_v2, simple_tag_v2, \
    simple_spread_v2, simple_reference_v2, simple_world_comm_v2, simple_speaker_listener_v3
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy, ComputeTDErrorMixin
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG as DDPG_CONFIG
from MPE.util.maddpg_tools import maddpg_actor_critic_loss, build_maddpg_models_and_action_dist, \
    maddpg_centralized_critic_postprocessing
from ray.rllib.agents.sac.sac_torch_policy import TargetNetworkMixin
from ray.tune.utils import merge_dicts
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG

from config_mpe import *
from MPE.model.torch_gru import *
from MPE.model.torch_gru_cc import *
from MPE.model.torch_lstm import *
from MPE.model.torch_lstm_cc import *
from MPE.model.torch_vd_ppo_a2c_gru_lstm import *
from MPE.util.mappo_tools import *
from MPE.util.maa2c_tools import *
from MPE.util.vda2c_tools import *
from MPE.util.vdppo_tools import *
from MPE.env.mpe_rllib import RllibMPE
from MPE.env.mpe_rllib_qmix import RllibMPE_QMIX

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

    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("GRU_ValueMixer", Torch_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("LSTM_ValueMixer", Torch_LSTM_Model_w_Mixer)

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

    register_env(args.map, lambda _: RllibMPE(env))

    test_env = RllibMPE(env)
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    n_agents = test_env.num_agents
    test_env.close()

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

        if args.continues:
            print(
                "{} do not support continue action space".format(args.run)
            )
            sys.exit()

        if args.map not in ["simple_spread", "simple_speaker_listener", "simple_reference"]:
            print(
                "adversarial agents contained in this MPE scenario. "
                "Not suitable for cooperative only algo {}".format(args.run)
            )
            sys.exit()

        if args.neural_arch not in ["GRU"]:
            print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
            sys.exit()

        if args.map == "simple_spread":
            env = simple_spread_v2.parallel_env(continuous_actions=False)
        elif args.map == "simple_reference":
            env = simple_reference_v2.parallel_env(continuous_actions=False)
        elif args.map == "simple_speaker_listener":
            env = simple_speaker_listener_v3.parallel_env(continuous_actions=False)


        test_env = RllibMPE_QMIX(env)
        agent_num = test_env.num_agents
        agent_list = test_env.agents
        obs_space = test_env.observation_space
        act_space = test_env.action_space
        test_env.close()

        obs_space = Tuple([obs_space] * agent_num)
        act_space = Tuple([act_space] * agent_num)

        # align with RWARE/env/rware_rllib_qmix.py reset() function in line 41-50
        grouping = {
            "group_1": [i for i in agent_list],
        }

        # QMIX/VDN algo needs grouping env
        register_env(
            args.map,
            lambda _: RllibMPE_QMIX(env).with_agent_groups(
                grouping, obs_space=obs_space, act_space=act_space))

        config = {
            "env": args.map,
            "train_batch_size": 32,
            "exploration_config": {
                "epsilon_timesteps": 5000,
                "final_epsilon": 0.05,
            },
            "mixer": "qmix" if args.run == "QMIX" else None,  # None for VDN, which has no mixer
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,
            "num_gpus_per_worker": args.num_gpus_per_worker,

        }

        results = tune.run("QMIX",
                           name=args.run + "_" + args.neural_arch + "_" + args.map,
                           stop=stop,
                           config=config,
                           verbose=1)

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

    elif args.run in ["SUM-VDA2C", "MIX-VDA2C"]:

        if args.map not in ["simple_spread", "simple_speaker_listener", "simple_reference"]:
            print(
                "adversarial agents contained in this MPE scenario. "
                "Not suitable for cooperative only algo {}".format(args.run)
            )
            sys.exit()

        config = {
            "model": {
                "custom_model": "{}_ValueMixer".format(args.neural_arch),
                "custom_model_config": {
                    "n_agents": n_agents,
                    "mixer": "qmix" if args.run == "MIX-VDA2C" else "vdn",
                    "mixer_emb_dim": 64,
                },
            },
        }
        config.update(common_config)

        VDA2C_CONFIG = merge_dicts(
            A2C_CONFIG,
            {
                "agent_num": n_agents,
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
            "num_sgd_iter": 10,
            "model": {
                "custom_model": "{}_ValueMixer".format(args.neural_arch),
                "custom_model_config": {
                    "n_agents": n_agents,
                    "mixer": "qmix" if args.run == "MIX-VDPPO" else "vdn",
                    "mixer_emb_dim": 64,
                },
            },
        }

        config.update(common_config)

        VDPPO_CONFIG = merge_dicts(
            PPO_CONFIG,
            {
                "agent_num": n_agents,
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

        from MPE.model.torch_maddpg import MADDPGTorchModel
        ModelCatalog.register_custom_model(
            "torch_maddpg", MADDPGTorchModel)

        config = {
            "model": {
                "custom_model": "torch_maddpg",
                "custom_model_config": {
                    "agent_num": n_agents
                },
            },
        }
        config.update(common_config)

        MADDPGTFPolicy = DDPGTFPolicy.with_updates(
            name="MADDPGTFPolicy",
            postprocess_fn=maddpg_centralized_critic_postprocessing,
            loss_fn=maddpg_actor_critic_loss,
            mixins=[
                TargetNetworkMixin,
                ComputeTDErrorMixin,
                CentralizedValueMixin
            ])

        MADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
            name="MADDPGTorchPolicy",
            get_default_config=lambda: DDPG_CONFIG,
            postprocess_fn=maddpg_centralized_critic_postprocessing,
            make_model_and_action_dist=build_maddpg_models_and_action_dist,
            loss_fn=maddpg_actor_critic_loss,
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

    elif args.run == "COMA":

        if args.continues:
            print("continues action space not supported in COMA")
            sys.exit()

        config = {
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "custom_model_config": {
                    "agent_num": n_agents,
                    "coma": True
                },
            },
        }
        config.update(common_config)

        from MPE.util.coma_tools import loss_with_central_critic_coma, central_vf_stats_coma, COMATorchPolicy

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

    ray.shutdown()
