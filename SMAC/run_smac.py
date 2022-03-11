# See also: centralized_critic.py for centralized critic PPO on twp step game.

import os
import ray

from gym.spaces import Dict as GymDict

from ray import tune
from ray.tune import register_env
from ray.tune.utils import merge_dicts

from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.env.multi_agent_env import ENV_STATE

from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.agents.dqn.r2d2 import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.dqn.r2d2_tf_policy import R2D2TFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin

from SMAC.model.torch_mask_lstm import *
from SMAC.model.torch_mask_lstm_cc import *
from SMAC.model.torch_mask_gru import *
from SMAC.model.torch_mask_gru_cc import *
from SMAC.model.torch_mask_updet import *
from SMAC.model.torch_mask_updet_cc import *
from SMAC.util.mappo_tools import *
from SMAC.util.maa2c_tools import *
from SMAC.metric.smac_callback import *
from SMAC.util.r2d2_tools import *
from SMAC.model.torch_qmix_mask_gru_updet import *
from SMAC.model.torch_vd_ppo_a2c_mask_gru_lstm_updet import *
from SMAC.util.vda2c_tools import value_mix_centralized_critic_postprocessing, value_mix_actor_critic_loss, \
    MixingValueMixin
from SMAC.util.vdppo_tools import value_mix_ppo_surrogate_loss
# from smac.env.starcraft2.starcraft2 import StarCraft2Env as SMAC
from SMAC.env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC


def run(args):
    # initialize env instance for env_info
    env = SMAC(map_name=args.map)
    env_info = env.get_env_info()
    state_shape = env_info["state_shape"]
    obs_shape = env_info["obs_shape"]
    n_actions = env_info["n_actions"]
    n_ally = env_info["n_agents"]
    n_enemy = env.death_tracker_enemy.shape[0]
    rollout_fragment_length = env_info["episode_limit"]
    env.close()
    # close env instance

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    register_env("smac", lambda config: SMAC(args.map))

    # Independent
    ModelCatalog.register_custom_model("LSTM_IndependentCritic", Torch_ActionMask_LSTM_Model)
    ModelCatalog.register_custom_model("GRU_IndependentCritic", Torch_ActionMask_GRU_Model)
    ModelCatalog.register_custom_model("UPDeT_IndependentCritic", Torch_ActionMask_Transformer_Model)

    # CTDE(centralized critic)
    ModelCatalog.register_custom_model("LSTM_CentralizedCritic",
                                       Torch_ActionMask_LSTM_CentralizedCritic_Model)
    ModelCatalog.register_custom_model("GRU_CentralizedCritic",
                                       Torch_ActionMask_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model("UPDeT_CentralizedCritic",
                                       Torch_ActionMask_Transformer_CentralizedCritic_Model)

    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("GRU_ValueMixer", Torch_ActionMask_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("LSTM_ValueMixer", Torch_ActionMask_LSTM_Model_w_Mixer)
    ModelCatalog.register_custom_model("UPDeT_ValueMixer", Torch_ActionMask_Transformer_Model_w_Mixer)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    common_config = {
        "num_gpus_per_worker": 0.2,
        "num_gpus": 0.6,
        "num_workers": 0,
        "train_batch_size": 1000,
        "env_config": {
            "map_name": args.map,
        },
        "multiagent": {
            "policies": {"shared_policy"},
            "policy_mapping_fn": (
                lambda agent_id, episode, **kwargs: "shared_policy"),
        },
        "callbacks": SmacCallbacks,
        "framework": args.framework,
    }

    if args.run in ["QMIX", "VDN"]:  # policy and model are implemented as source code is

        if args.neural_arch not in ["GRU", "UPDeT"]:
            assert NotImplementedError

        grouping = {
            "group_1": [i for i in range(n_ally)],
        }
        ## obs state setting here
        obs_space = Tuple([
                              GymDict({
                                  "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
                                  ENV_STATE: Box(-2.0, 2.0, shape=(state_shape,)),
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
            "env": "grouped_smac",
            "env_config": {
                "map_name": args.map,
            },
            "rollout_fragment_length": rollout_fragment_length,
            "train_batch_size": 400,
            "exploration_config": {
                "epsilon_timesteps": 5000,
                "final_epsilon": 0.05,
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

            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
            "num_workers": 2,
        }

        results = tune.run(QMixTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config, verbose=1)

    elif args.run in ["SUM-VDA2C", "MIX-VDA2C"]:

        config = {
            "env": SMAC,
            "model": {
                "custom_model": "{}_ValueMixer".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "token_dim": args.token_dim,
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape,
                    "mixer": "qmix" if args.run == "MIX-VDA2C" else "vdn",
                    "mixer_emb_dim": 64,
                },
            },
        }
        config.update(common_config)

        VDA2C_CONFIG = merge_dicts(
            A2C_CONFIG,
            {
                "agent_num": n_ally,
                "state_dim": state_shape,
                "self_obs_dim": obs_shape,
            }
        )

        VDA2C_CONFIG["rollout_fragment_length"] = rollout_fragment_length

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
        sgd_minibatch_size = 128
        while sgd_minibatch_size < rollout_fragment_length:
            sgd_minibatch_size *= 2

        config = {
            "env": SMAC,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": 10,
            "model": {
                "custom_model": "{}_ValueMixer".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "token_dim": args.token_dim,
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape,
                    "mixer": "qmix" if args.run == "MIX-VDPPO" else "vdn",
                    "mixer_emb_dim": 64,
                },
            },
        }
        config.update(common_config)

        VDPPO_CONFIG = merge_dicts(
            PPO_CONFIG,
            {
                "agent_num": n_ally,
                "state_dim": state_shape,
                "self_obs_dim": obs_shape,
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


    elif args.run in ["R2D2"]:  # similar to IQL in recurrent/POMDP mode

        # ray built-in Q series algo is not very flexible
        if args.neural_arch not in ["GRU", "LSTM"]:
            assert NotImplementedError

        config = {
            "num_gpus_per_worker": 0.2,
            "num_gpus": 0.6,
            "num_workers": 0,
            "train_batch_size": 1000,
            "env": SMAC,
            "env_config": {
                "map_name": args.map,
            },
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape
                },
            },
            "callbacks": SmacCallbacks,
            "framework": args.framework,
        }

        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return R2D2WithMaskPolicy

        DEFAULT_CONFIG['dueling'] = False
        R2D2WithMaskTrainer = build_trainer(
            name="R2D2_Trainer",
            default_config=DEFAULT_CONFIG,
            default_policy=R2D2TFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(R2D2WithMaskTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config,
                           verbose=1)

    elif args.run in ["PG", "A2C", "A3C"]:  # PG need define action mask GRU / only torch now

        config = {
            "env": SMAC,
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "token_dim": args.token_dim,
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape
                },
            },
        }
        config.update(common_config)
        results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                           verbose=1)

    elif args.run == "MAA2C":  # centralized A2C

        config = {
            "env": SMAC,
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "token_dim": args.token_dim,
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape
                },
            },
        }
        config.update(common_config)

        MAA2C_CONFIG = merge_dicts(
            A2C_CONFIG,
            {
                "agent_num": n_ally,
                "state_dim": state_shape,
                "self_obs_dim": obs_shape,
                "centralized_critic_obs_dim": -1,
            }
        )

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
            get_default_config=lambda: MAA2C_CONFIG,
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

        results = tune.run(MAA2CTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config, verbose=1)

    elif args.run in ["PPO", "APPO"]:

        """
        for bug mentioned https://github.com/ray-project/ray/pull/20743
        make sure sgd_minibatch_size > max_seq_len
        """
        sgd_minibatch_size = 128
        while sgd_minibatch_size < rollout_fragment_length:
            sgd_minibatch_size *= 2

        config = {
            "env": SMAC,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": sgd_minibatch_size,
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "token_dim": args.token_dim,
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape
                },
            },
        }
        config.update(common_config)
        results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                           verbose=1)

    elif args.run == "MAPPO":  # centralized PPO

        """
        for bug mentioned https://github.com/ray-project/ray/pull/20743
        make sure sgd_minibatch_size > max_seq_len
        """
        sgd_minibatch_size = 128
        while sgd_minibatch_size < rollout_fragment_length:
            sgd_minibatch_size *= 2

        config = {
            "env": SMAC,
            "num_sgd_iter": 10,
            "sgd_minibatch_size": sgd_minibatch_size,
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "token_dim": args.token_dim,
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape
                },
            },
        }
        config.update(common_config)

        MAPPO_CONFIG = merge_dicts(
            PPO_CONFIG,
            {
                "agent_num": n_ally,
                "state_dim": state_shape,
                "self_obs_dim": obs_shape,
                "centralized_critic_obs_dim": -1,
            }
        )

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
            get_default_config=lambda: MAPPO_CONFIG,
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

        results = tune.run(MAPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config,
                           verbose=1)
    elif args.run == "COMA":

        config = {
            "env": SMAC,
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "max_seq_len": rollout_fragment_length,
                "custom_model_config": {
                    "token_dim": args.token_dim,
                    "ally_num": n_ally,
                    "enemy_num": n_enemy,
                    "self_obs_dim": obs_shape,
                    "state_dim": state_shape,
                    "coma": True
                },
            },
        }

        config.update(common_config)

        from SMAC.util.coma_tools import loss_with_central_critic_coma, central_vf_stats_coma, COMATorchPolicy

        COMA_CONFIG = merge_dicts(
            A2C_CONFIG,
            {
                "agent_num": n_ally,
                "state_dim": state_shape,
                "self_obs_dim": obs_shape,
                "centralized_critic_obs_dim": -1,
            }
        )

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
            get_default_config=lambda: COMA_CONFIG,
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
                 name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                 verbose=1)

    else:
        print("args.run illegal")
        raise ValueError()

    # if args.as_test:
    #     config["seed"] = 1234

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
