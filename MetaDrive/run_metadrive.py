from ray.tune.registry import register_env

import sys
import ray
from ray.tune.registry import _global_registry, ENV_CREATOR
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer

from MetaDrive.model.torch_gru import Torch_GRU_Model
from MetaDrive.model.torch_lstm import Torch_LSTM_Model
from MetaDrive.model.torch_gru_cc import Torch_GRU_CentralizedCritic_Model
from MetaDrive.model.torch_lstm_cc import Torch_LSTM_CentralizedCritic_Model
from MetaDrive.util.mappo_tools import *
from MetaDrive.util.maa2c_tools import *
from MetaDrive.config_metadrive import *
from MetaDrive.env.meta_drive_rllib import *

if __name__ == "__main__":
    args = get_train_parser().parse_args()

    ray.init(local_mode=True)

    # env setup
    if args.map == "Bottleneck":
        Env_Class = Bottleneck_RLlib_Centralized_Critic if "MA" in args.run else Bottleneck_RLlib
    elif args.map == "ParkingLot":
        Env_Class = ParkingLot_RLlib_Centralized_Critic if "MA" in args.run else ParkingLot_RLlib
    elif args.map == "Intersection":
        Env_Class = Intersection_RLlib_Centralized_Critic if "MA" in args.run else Intersection_RLlib
    elif args.map == "Roundabout":
        Env_Class = Roundabout_RLlib_Centralized_Critic if "MA" in args.run else Roundabout_RLlib
    elif args.map == "Tollgate":
        Env_Class = Tollgate_RLlib_Centralized_Critic if "MA" in args.run else Tollgate_RLlib
    else:
        sys.exit()

    register_env(args.map, lambda c: Env_Class(c))

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

    env_config = {
        "start_seed": 1234,
        "num_agents": 5,
        "crash_done": True
    }

    # create an env to get obs act space for multi-agent policy specify
    env_class = _global_registry.get(ENV_CREATOR, args.map)
    single_env = env_class(env_config)
    obs_space = single_env.observation_space["agent0"]
    act_space = single_env.action_space["agent0"]

    common_config = {
        "env": args.map,
        "env_config": env_config,
        "num_gpus": args.num_gpus,
        "num_workers": 1,
        "train_batch_size": 1000,
        "multiagent": {
            "policies": {
                "shared_policy": (None, obs_space, act_space, {})
            },
            "policy_mapping_fn": lambda x: "shared_policy"
        },
        "framework": args.framework,

    }

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    if args.run in ["QMIX", "VDN"]:
        print(
            "\nAdversarial agents contained in Meta-Drive. "
            "Not suitable for cooperative only algo like {}".format(args.run)
        )
        raise ValueError()

    elif args.run in ["R2D2"]:
        print(
            "\nAction space of Meta-Drive (Box) is not supported for Q function "
            "based algo like DQN and R2D2 in Ray/RLlib, only Discrete supported"
        )
        raise ValueError()


    elif args.run in ["A3C"]:
        print(
            "\nCurrently not support A3C for Meta-Drive, as A3C creates multiple envs in one process, which is "
            "illegal in Meta-Drive"
        )
        raise ValueError()

    elif args.run in ["A2C", "PG"]:

        config = {
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
        }

        config.update(common_config)

        tune.run(args.run,
                 name=args.run + "_" + args.neural_arch + "_" + args.map,
                 stop=stop,
                 config=config,
                 verbose=1)

    elif args.run in ["MAA2C"]:

        config = {
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            }
        }

        config.update(common_config)


        def get_policy_class(config):
            if config["framework"] == "torch":
                return MAA2CTorchPolicy
            else:
                raise ValueError()


        MAA2C_CONFIG = {
            "real_parameter_sharing": True,
            "counterfactual": False,
            "centralized_critic_obs_dim": -1,
            "num_neighbours": args.num_neighbours,
            "framework": "torch",
            "fuse_mode": "mf",  # In ["concat", "mf"]
            "mf_nei_distance": 10,
        }

        centralized_critic_obs_dim = get_centralized_critic_obs_dim(
            obs_space, act_space, MAA2C_CONFIG["counterfactual"], MAA2C_CONFIG["num_neighbours"],
            MAA2C_CONFIG["fuse_mode"]
        )

        MAA2C_CONFIG.update(A2C_CONFIG)

        MAA2C_CONFIG["centralized_critic_obs_dim"] = centralized_critic_obs_dim

        MAA2CTorchPolicy = A3CTorchPolicy.with_updates(
            name="MAA2CTorchPolicy",
            get_default_config=lambda: MAA2C_CONFIG,
            make_model=make_model,
            extra_action_out_fn=vf_preds_fetches,
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic_a2c,
            stats_fn=central_vf_stats_a2c,
            mixins=[CentralizedValueMixin]
        )

        MAA2CTrainer = A2CTrainer.with_updates(
            name="MAA2CTrainer",
            default_config=MAA2C_CONFIG,
            default_policy=MAA2CTorchPolicy,
            get_policy_class=get_policy_class,
        )

        tune.run(MAA2CTrainer,
                 name=args.run + "_" + args.neural_arch + "_" + args.map,
                 stop=stop,
                 config=config,
                 verbose=1)

    elif args.run == "DDPG":

        tune.run(
            args.run,
            name=args.run + "_" + args.neural_arch + "_" + args.map,
            stop=stop,
            config=common_config,
            verbose=1
        )

    elif args.run == "MADDPG":

        from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
        from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy, ComputeTDErrorMixin
        from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
        from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG as DDPG_CONFIG
        from MetaDrive.util.maddpg_tools import loss_with_central_critic_ddpg
        from ray.rllib.agents.sac.sac_torch_policy import TargetNetworkMixin
        from MetaDrive.model.torch_maddpg import DDPGCentralizedCriticModel

        ModelCatalog.register_custom_model(
            "torch_maddpg", DDPGCentralizedCriticModel)

        MADDPG_CONFIG = {
            "real_parameter_sharing": True,
            "counterfactual": False,
            "centralized_critic_obs_dim": -1,
            "num_neighbours": args.num_neighbours,
            "framework": "torch",
            "fuse_mode": "mf",  # In ["concat", "mf"]
            "mf_nei_distance": 10,
        }

        centralized_critic_obs_dim = get_centralized_critic_obs_dim(
            obs_space, act_space, MADDPG_CONFIG["counterfactual"], MADDPG_CONFIG["num_neighbours"],
            MADDPG_CONFIG["fuse_mode"]
        )

        config = {
            "model": {
                "custom_model": "torch_maddpg",
                "custom_model_config": {
                    "centralized_critic_obs_dim": centralized_critic_obs_dim
                },
            },
        }
        config.update(common_config)


        MADDPG_CONFIG.update(DDPG_CONFIG)

        MADDPG_CONFIG["centralized_critic_obs_dim"] = centralized_critic_obs_dim

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
            get_default_config=lambda: MADDPG_CONFIG,
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
            default_config=MADDPG_CONFIG,
            default_policy=MADDPGTorchPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(MADDPGTrainer,
                           name=args.run + "_" + args.map,
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

        tune.run(args.run,
                 name=args.run + "_" + args.neural_arch + "_" + args.map,
                 stop=stop,
                 config=config,
                 verbose=1)

    elif args.run in ["MAPPO"]:

        config = {
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            },
            "num_sgd_iter": 10,
        }
        config.update(common_config)


        def get_policy_class(config):
            if config["framework"] == "torch":
                return MAPPOTorchPolicy
            else:
                raise ValueError()


        MAPPO_CONFIG = {
            "real_parameter_sharing": True,
            "counterfactual": False,
            "centralized_critic_obs_dim": -1,
            "num_neighbours": args.num_neighbours,
            "framework": "torch",
            "fuse_mode": "mf",  # In ["concat", "mf"]
            "mf_nei_distance": 10,
        }

        centralized_critic_obs_dim = get_centralized_critic_obs_dim(
            obs_space, act_space, MAPPO_CONFIG["counterfactual"], MAPPO_CONFIG["num_neighbours"],
            MAPPO_CONFIG["fuse_mode"]
        )

        MAPPO_CONFIG.update(PPO_CONFIG)

        MAPPO_CONFIG["centralized_critic_obs_dim"] = centralized_critic_obs_dim

        MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
            name="MAPPOTorchPolicy",
            get_default_config=lambda: MAPPO_CONFIG,
            make_model=make_model,
            extra_action_out_fn=vf_preds_fetches,
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic_ppo,
            stats_fn=central_vf_stats_ppo,
            before_init=setup_torch_mixins,
            mixins=[TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin, CentralizedValueMixin]
        )

        MAPPOTrainer = PPOTrainer.with_updates(
            name="MAPPOTrainer",
            default_config=MAPPO_CONFIG,
            default_policy=MAPPOTorchPolicy,
            get_policy_class=get_policy_class,
        )

        tune.run(MAPPOTrainer,
                 name=args.run + "_" + args.neural_arch + "_" + args.map,
                 stop=stop,
                 config=config,
                 verbose=1)
