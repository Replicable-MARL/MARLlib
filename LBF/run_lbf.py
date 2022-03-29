from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
import ray
from ray import tune
from ray.tune.registry import register_env

from LBF.config_lbf import get_train_parser
from LBF.env.lbf_rllib import RllibLBF

from LBF.model.torch_gru import *
from LBF.model.torch_gru_cc import *
from LBF.model.torch_lstm import *
from LBF.model.torch_lstm_cc import *

from LBF.model.torch_vd_ppo_a2c_gru_lstm import *
from LBF.util.vdppo_tools import *

from LBF.policy.pg_a2c_a3c_r2d2 import run_pg_a2c_a3c_r2d2
from LBF.policy.vdn_qmix import run_vdn_qmix
from LBF.policy.ppo import run_ppo
from LBF.policy.vda2c import run_vda2c_sum_mix
from LBF.policy.vdppo import run_vdppo_sum_mix
from LBF.policy.maa2c import run_maa2c
from LBF.policy.mappo import run_mappo
from LBF.policy.coma import run_coma

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    agent_num = args.agent_num

    env_config = {
        "num_agents": args.agent_num,
        "field_size": args.field_size,
        "max_food": args.max_food,
        "sight": args.sight,
        "force_coop": args.force_coop,
    }

    map_name = "Foraging-{4}s-{0}x{0}-{1}p-{2}f{3}".format(
        args.field_size,
        args.agent_num,
        args.max_food,
        "-coop" if args.force_coop else "",
        args.sight
    )

    register_env("lbf", lambda _: RllibLBF(env_config))

    single_env = RllibLBF(env_config)
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
        "VDN": run_vdn_qmix,
        "QMIX": run_vdn_qmix,
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
        "env": "lbf",
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
