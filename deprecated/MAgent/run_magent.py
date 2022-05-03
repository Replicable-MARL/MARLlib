from ray.rllib.models import ModelCatalog
import sys
from ray.tune import register_env
from ray import tune
import ray

from ray.rllib.utils.test_utils import check_learning_achieved
from pettingzoo.magent import adversarial_pursuit_v3, battle_v3, battlefield_v3, combined_arms_v5, gather_v3, \
    tiger_deer_v3
from MAgent.config_magent import get_train_parser

from MAgent.model.torch_cnn_lstm import Torch_CNN_LSTM_Model
from MAgent.model.torch_cnn_lstm_cc import Torch_CNN_LSTM_CentralizedCritic_Model
from MAgent.model.torch_cnn_gru import Torch_CNN_GRU_Model
from MAgent.model.torch_cnn_gru_cc import Torch_CNN_GRU_CentralizedCritic_Model
# from MPE.util.vda2c_tools import *
# from MPE.util.vdppo_tools import *
from MAgent.env.magent_rllib import RllibMAgent

from MAgent.policy.pg_a2c_a3c_r2d2 import run_pg_a2c_a3c_r2d2
from MAgent.policy.ppo import run_ppo
from MAgent.policy.maa2c import run_maa2c
from MAgent.policy.mappo import run_mappo
from MAgent.policy.coma import run_coma

# tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    if args.run not in ["R2D2", "PG", "A2C", "A3C", "PPO"]:
        minimap_mode = True
    else:
        minimap_mode = False

    if args.map == "adversarial_pursuit":
        env = adversarial_pursuit_v3.env(minimap_mode=minimap_mode)
        mini_channel_dim = 4
    elif args.map == "battle":
        env = battle_v3.env(minimap_mode=minimap_mode)
        mini_channel_dim = 4
    elif args.map == "battlefield":
        env = battlefield_v3.env(minimap_mode=minimap_mode)
        mini_channel_dim = 4
    elif args.map == "combined_arms":
        env = combined_arms_v5.env(minimap_mode=minimap_mode)
        mini_channel_dim = 6
    elif args.map == "gather":
        env = gather_v3.env(minimap_mode=minimap_mode)
        mini_channel_dim = 4
    elif args.map == "tiger_deer":
        env = tiger_deer_v3.env(minimap_mode=minimap_mode)
        mini_channel_dim = 6

    else:
        assert NotImplementedError
        print("Scenario {} not exists in pettingzoo".format(args.map))
        sys.exit()

    register_env(args.map, lambda _: RllibMAgent(env))

    test_env = RllibMAgent(env)
    obs = test_env.reset()
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    n_agents = test_env.num_agents
    agents_name = test_env.agents

    agents_type = set()
    for agent_name in agents_name:
        agent_type_name = agent_name.split("_")[0]
        agents_type.add(agent_type_name)

    env_config = {
        "n_agents": n_agents,
        "mini_channel_dim": mini_channel_dim
    }

    ##############
    ### policy ###
    ##############

    if args.share_policy:  # fully_shared
        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")
    else:  # partly_shared
        policies = {
            "{}_policy".format(agent_type): (None, obs_space, act_space, {}) for agent_type in agents_type
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = lambda policy_id: "{}_policy".format(policy_id.split("_")[0])

    policy_function_dict = {
        "PG": run_pg_a2c_a3c_r2d2,
        "A2C": run_pg_a2c_a3c_r2d2,
        "A3C": run_pg_a2c_a3c_r2d2,
        "R2D2": run_pg_a2c_a3c_r2d2,
        "PPO": run_ppo,
        "MAA2C": run_maa2c,
        "MAPPO": run_mappo,
        "COMA": run_coma,
    }

    #############
    ### model ###
    #############

    # Independent
    ModelCatalog.register_custom_model("CNN_LSTM", Torch_CNN_LSTM_Model)
    ModelCatalog.register_custom_model("CNN_GRU", Torch_CNN_GRU_Model)

    # CTDE(centralized critic (only action))
    ModelCatalog.register_custom_model(
        "CNN_GRU_CentralizedCritic", Torch_CNN_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "CNN_LSTM_CentralizedCritic", Torch_CNN_LSTM_CentralizedCritic_Model)

    #####################
    ### common config ###
    #####################

    common_config = {
        "env": args.map,
        "num_gpus_per_worker": 0.2,
        "num_gpus": 0.6,
        "num_workers": 0,
        "train_batch_size": 1000,
        "rollout_fragment_length": 30,
        "horizon": 200,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": args.framework,
    }

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    ##################
    ### run script ###
    ###################

    results = policy_function_dict[args.run](args, common_config, env_config, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
