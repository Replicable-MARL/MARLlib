# See also: centralized_critic.py for centralized critic PPO on twp step game.
import sys

from gym.spaces import Dict as GymDict
from ray import tune
from ray.tune import register_env
from ray.tune.utils import merge_dicts

from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.agents.dqn.r2d2 import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.dqn.r2d2_tf_policy import R2D2TFPolicy

from model.torch_mask_lstm import *
from model.torch_mask_lstm_cc import *
from model.torch_mask_gru import *
from model.torch_mask_gru_cc import *
from model.torch_mask_updet import *
from model.torch_mask_updet_cc import *
from cc_util.mappo_tools import *
from cc_util.maa2c_tools import *
from metric.smac_callback import *
from model.torch_mask_r2d2 import *
from model.torch_qmix import *

# from smac.env.starcraft2.starcraft2 import StarCraft2Env as SMAC
from env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC


def run_parallel(args):
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

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    common_config = {
        "num_gpus_per_worker": args.num_gpus_per_worker,
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "train_batch_size": tune.grid_search([1000, 2000, 4000]),
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
        "seed": tune.grid_search([i * + 0 for i in range(args.num_seeds)]),
    }

    if args.run in ["VDN", "QMIX"]:  # policy are implemented as source code is

        if args.neural_arch not in ["GRU", "UPDeT"]:
            assert NotImplementedError
            print(args.neural_arch + " is not supported in " + args.run)
            sys.exit()

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
            "train_batch_size": tune.grid_search([1000, 2000]),
            "exploration_config": {
                "epsilon_timesteps": tune.grid_search([10000, 20000, 50000]),
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
            "num_gpus": args.num_gpus,
            "num_gpus_per_worker": args.num_gpus_per_worker,
            "num_workers": args.num_workers,
            "seed": tune.grid_search([i * 100 + 0 for i in range(args.num_seeds)])
        }

        results = tune.run(QMixTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config, verbose=1)


    elif args.run in ["R2D2"]:  # similar to IQL in recurrent/POMDP mode

        # ray built-in Q series algo is not very flexible for transformer-like model
        if args.neural_arch not in ["GRU", "LSTM"]:
            assert NotImplementedError
            print(args.neural_arch + " is not supported in " + args.run)
            sys.exit()

        config = {
            "num_gpus_per_worker": args.num_gpus_per_worker,
            "num_gpus": args.num_gpus,
            "num_workers": args.num_workers,
            "train_batch_size": tune.grid_search([1000, 2000]),
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

        DEFAULT_CONFIG['dueling'] = False  # only support False | default True in dqn

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
        results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config, verbose=1)

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
            A3C_CONFIG,
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

        config = {
            "env": SMAC,
            "num_sgd_iter": tune.grid_search([5, 10, 20]),
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
        results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config, verbose=1)

    elif args.run == "MAPPO":  # centralized PPO

        config = {
            "env": SMAC,
            "num_sgd_iter": tune.grid_search([5, 10, 20]),
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

    else:
        assert NotImplementedError
        print(args.run + " algo is not Implemented")
        sys.exit()

    # if args.as_test:
    #     config["seed"] = 1234

    # if args.as_test:
    #     check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
