from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved

from ray import tune
from ray.tune.registry import register_env

from RWARE.config_rware import get_train_parser
from RWARE.env.rware_rllib import RllibRWARE

from RWARE.model.torch_gru import *
from RWARE.model.torch_gru_cc import *
from RWARE.model.torch_lstm import *
from RWARE.model.torch_lstm_cc import *
from RWARE.model.torch_vd_ppo_a2c_gru_lstm import *
from RWARE.util.vdppo_tools import *

from RWARE.policy.pg_a2c_a3c_r2d2 import run_pg_a2c_a3c_r2d2
from RWARE.policy.vdn_qmix_iql import run_vdn_qmix_iql
from RWARE.policy.ppo import run_ppo
from RWARE.policy.vda2c import run_vda2c_sum_mix
from RWARE.policy.vdppo import run_vdppo_sum_mix
from RWARE.policy.maa2c import run_maa2c
from RWARE.policy.mappo import run_mappo
from RWARE.policy.coma import run_coma

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    agent_num = args.agents_num

    env_config = {
        "agents_num": args.agents_num,
        "map_size": args.map_size,
        "difficulty": args.difficulty,
    }

    map_name = "rware-{0}-{1}ag-{2}".format(
        args.map_size,
        args.agents_num,
        args.difficulty,
    )

    register_env("rware", lambda _: RllibRWARE(env_config))

    single_env = RllibRWARE(env_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    ##############
    ### policy ###
    ##############

    if args.share_policy:
        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")
    else:
        policies = {
            "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(agent_num)
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[int(agent_id[6:])])

    policy_function_dict = {
        "PG": run_pg_a2c_a3c_r2d2,
        "A2C": run_pg_a2c_a3c_r2d2,
        "A3C": run_pg_a2c_a3c_r2d2,
        "R2D2": run_pg_a2c_a3c_r2d2,
        "IQL": run_vdn_qmix_iql,
        "VDN": run_vdn_qmix_iql,
        "QMIX": run_vdn_qmix_iql,
        "PPO": run_ppo,
        "MIX-VDA2C": run_vda2c_sum_mix,
        "SUM-VDA2C": run_vda2c_sum_mix,
        "MIX-VDPPO": run_vdppo_sum_mix,
        "SUM-VDPPO": run_vdppo_sum_mix,
        "MAA2C": run_maa2c,
        "MAPPO": run_mappo,
        "COMA": run_coma,
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

    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("GRU_ValueMixer", Torch_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("LSTM_ValueMixer", Torch_LSTM_Model_w_Mixer)

    #####################
    ### common config ###
    #####################

    common_config = {
        "seed": 1,
        "env": "rware",
        "num_gpus_per_worker": args.num_gpus_per_worker,
        "train_batch_size": 1000,
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
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

    results = policy_function_dict[args.run](args, common_config, env_config, map_name, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
