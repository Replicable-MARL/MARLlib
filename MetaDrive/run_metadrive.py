from ray.tune.registry import register_env

import sys
import ray
from ray.tune.registry import _global_registry, ENV_CREATOR

from ray.rllib.utils.test_utils import check_learning_achieved
from MetaDrive.model.torch_gru import Torch_GRU_Model
from MetaDrive.model.torch_lstm import Torch_LSTM_Model
from MetaDrive.model.torch_gru_cc import Torch_GRU_CentralizedCritic_Model
from MetaDrive.model.torch_lstm_cc import Torch_LSTM_CentralizedCritic_Model
from MetaDrive.util.maa2c_tools import *
from MetaDrive.config_metadrive import *
from MetaDrive.env.meta_drive_rllib import *

from MetaDrive.policy.pg_a2c import run_pg_a2c
from MetaDrive.policy.ppo import run_ppo
from MetaDrive.policy.maa2c import run_maa2c
from MetaDrive.policy.mappo import run_mappo
from MetaDrive.policy.ddpg import run_ddpg
from MetaDrive.policy.maddpg import run_maddpg

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

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

    env_config = {
        "start_seed": 0,
        "num_agents": args.num_agents,
        "crash_done": True
    }

    env_class = _global_registry.get(ENV_CREATOR, args.map)
    single_env = env_class(env_config)
    obs_space = single_env.observation_space["agent0"]
    act_space = single_env.action_space["agent0"]


    ##############
    ### policy ###
    ##############

    policies = {
        "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(args.num_agents)
    }
    policy_ids = list(policies.keys())


    policy_function_dict = {
        "PG": run_pg_a2c,
        "A2C": run_pg_a2c,
        "PPO": run_ppo,
        "MAA2C": run_maa2c,
        "MAPPO": run_mappo,
        "DDPG": run_ddpg,
        "MADDPG": run_maddpg,
    }


    #############
    ### model ###
    #############

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


    #####################
    ### common config ###
    #####################

    ma_config = {
        "real_parameter_sharing": True,
        "counterfactual": False,
        "centralized_critic_obs_dim": -1,
        "num_neighbours": args.num_neighbours,
        "framework": "torch",
        "fuse_mode": args.fuse_mode,  # In ["concat", "mf"]
        "mf_nei_distance": 10,
    }

    common_config = {
        "env": args.map,
        "env_config": env_config,
        "num_gpus": args.num_gpus,
        "num_workers": 1 if args.local_mode else args.num_workers,
        "train_batch_size": 1000,
        "multiagent": {
            "policies": {
                "shared_policy": (None, obs_space, act_space, {})
            },
            "policy_mapping_fn": lambda x: "shared_policy"
        },
        "framework": args.framework,

    }

    cc_obs_dim = get_centralized_critic_obs_dim(
        obs_space, act_space, ma_config["counterfactual"], ma_config["num_neighbours"],
        ma_config["fuse_mode"]
    )

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    ##################
    ### run script ###
    ###################

    results = policy_function_dict[args.run](args, common_config, ma_config, cc_obs_dim, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()


