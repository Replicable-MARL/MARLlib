from ray.rllib.env import PettingZooEnv
from ray import tune
from ray.tune import register_env
from ray.rllib.utils.test_utils import check_learning_achieved
from pettingzoo.classic import hanabi_v4

from config_hanabi import *
from Hanabi.model.torch_mask_gru import *
from Hanabi.model.torch_mask_gru_cc import *
from Hanabi.model.torch_mask_lstm import *
from Hanabi.model.torch_mask_lstm_cc import *
from Hanabi.model.torch_mask_r2d2 import *
from Hanabi.util.maa2c_tools import *
from Hanabi.policy.pg_a2c_a3c import run_pg_a2c_a3c
from Hanabi.policy.r2d2 import run_r2d2
from Hanabi.policy.ppo import run_ppo
from Hanabi.policy.maa2c import run_maa2c
from Hanabi.policy.mappo import run_mappo
from Hanabi.policy.coma import run_coma

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    agent_num = args.num_players
    env = hanabi_v4.env(players=agent_num)

    register_env("Hanabi", lambda _: PettingZooEnv(env))

    test_env = PettingZooEnv(env)
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    n_agents = len(test_env.agents)

    ##############
    ### policy ###
    ##############

    if args.share_policy:
        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")
    else:
        policies = {
            "player_{}".format(i): (None, obs_space, act_space, {}) for i in range(agent_num)
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = lambda agent_id: agent_id

    policy_function_dict = {
        "PG": run_pg_a2c_a3c,
        "A2C": run_pg_a2c_a3c,
        "A3C": run_pg_a2c_a3c,
        "R2D2": run_r2d2,
        "PPO": run_ppo,
        "MAA2C": run_maa2c,
        "MAPPO": run_mappo,
        "COMA": run_coma,
    }

    #############
    ### model ###
    #############

    # Independent
    ModelCatalog.register_custom_model(
        "GRU_IndependentCritic", Torch_ActionMask_GRU_Model)
    ModelCatalog.register_custom_model(
        "LSTM_IndependentCritic", Torch_ActionMask_LSTM_Model)

    # CTDE(centralized critic)
    ModelCatalog.register_custom_model(
        "GRU_CentralizedCritic", Torch_ActionMask_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "LSTM_CentralizedCritic", Torch_ActionMask_LSTM_CentralizedCritic_Model)


    #####################
    ### common config ###
    #####################

    common_config = {
        "env": "Hanabi",
        "num_gpus_per_worker": args.num_gpus_per_worker,
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
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

    results = policy_function_dict[args.run](args, common_config, n_agents, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
