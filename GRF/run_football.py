from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
import ray
from ray import tune
from ray.tune.registry import register_env

from GRF.config_football import get_train_parser
from GRF.env.football_rllib import RllibGFootball
from GRF.model.torch_cnn_lstm import Torch_CNN_LSTM_Model
from GRF.model.torch_cnn_lstm_cc import Torch_CNN_LSTM_CentralizedCritic_Model
from GRF.model.torch_cnn_gru import Torch_CNN_GRU_Model
from GRF.model.torch_cnn_gru_cc import Torch_CNN_GRU_CentralizedCritic_Model
from GRF.model.torch_cnn_updet import Torch_CNN_Transformer_Model
from GRF.model.torch_cnn_updet_cc import Torch_CNN_Transformer_CentralizedCritic_Model
from GRF.model.torch_vda2c_vdppo_model import *
from GRF.util.vdppo_tools import *
from GRF.policy.pg_a2c_a3c_r2d2 import run_pg_a2c_a3c_r2d2
from GRF.policy.vdn_qmix_iql import run_vdn_qmix_iql
from GRF.policy.ppo import run_ppo
from GRF.policy.vda2c import run_vda2c_sum_mix
from GRF.policy.vdppo import run_vdppo_sum_mix
from GRF.policy.maa2c import run_maa2c
from GRF.policy.mappo import run_mappo
from GRF.policy.coma import run_coma

ally_num_dict = {
    "academy_empty_goal": 2,
    "academy_empty_goal_close": 2,
    "academy_run_to_score_with_keeper": 2,
    "academy_run_to_score": 2,
    "academy_pass_and_shoot_with_keeper": 3,
    "academy_run_pass_and_shoot_with_keeper": 3,
    "academy_3_vs_1_with_keeper": 4,
    "academy_single_goal_versus_lazy": 11,
    "academy_corner": 11,
    "academy_counterattack_easy": 11,
    "academy_counterattack_hard": 11,
}

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    ally_num = ally_num_dict[args.map]

    env_config = {
        "env_name": args.map,
        "num_agents": ally_num
    }

    register_env("football", lambda _: RllibGFootball(env_config))

    single_env = RllibGFootball(env_config)
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
            "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(ally_num)
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
    ModelCatalog.register_custom_model("CNN_LSTM", Torch_CNN_LSTM_Model)
    ModelCatalog.register_custom_model("CNN_GRU", Torch_CNN_GRU_Model)
    ModelCatalog.register_custom_model("CNN_UPDeT", Torch_CNN_Transformer_Model)

    # CTDE(centralized critic (only action))
    ModelCatalog.register_custom_model(
        "CNN_GRU_CentralizedCritic", Torch_CNN_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "CNN_LSTM_CentralizedCritic", Torch_CNN_LSTM_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "CNN_UPDeT_CentralizedCritic", Torch_CNN_Transformer_CentralizedCritic_Model)
    #
    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("CNN_GRU_ValueMixer", Torch_CNN_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("CNN_LSTM_ValueMixer", Torch_CNN_LSTM_Model_w_Mixer)
    ModelCatalog.register_custom_model("CNN_UPDeT_ValueMixer", Torch_CNN_Transformer_Model_w_Mixer)

    #####################
    ### common config ###
    #####################

    common_config = {
        "seed": 1,
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
    ##################

    results = policy_function_dict[args.run](args, common_config, env_config, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
