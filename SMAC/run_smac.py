from ray import tune
from ray.tune import register_env
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.utils import merge_dicts
import random
from SMAC.model.torch_mask_lstm import *
from SMAC.model.torch_mask_lstm_cc import *
from SMAC.model.torch_mask_gru import *
from SMAC.model.torch_mask_gru_cc import *
from SMAC.model.torch_mask_updet_cc import *
from SMAC.metric.smac_callback import *
from SMAC.util.r2d2_tools import *
from SMAC.model.torch_vdn_qmix_iql_model import *
from SMAC.model.torch_vda2c_vdppo_model import *
from SMAC.env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC
from config_smac import *
from SMAC.policy.vdn_qmix_iql import run_vdn_qmix_iql
from SMAC.policy.pg_a2c_a3c import run_pg_a2c_a3c
from SMAC.policy.r2d2 import run_r2d2
from SMAC.policy.ppo import run_ppo
from SMAC.policy.vda2c import run_vda2c_sum_mix
from SMAC.policy.vdppo import run_vdppo_sum_mix
from SMAC.policy.maa2c import run_maa2c
from SMAC.policy.mappo import run_mappo
from SMAC.policy.coma import run_coma
from SMAC.metric.smac_logger import SMACLogger
from SMAC.metric.smac_reporter import SMACReporter

if __name__ == '__main__':

    args = get_train_parser().parse_args()
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode, log_to_driver=False)

    ###################
    ### environment ###
    ###################

    env = SMAC(map_name=args.map)
    env_info = env.get_env_info()
    obs_space = env.observation_space
    act_space = env.action_space
    state_shape = env_info["state_shape"]
    obs_shape = env_info["obs_shape"]
    n_actions = env_info["n_actions"]
    n_ally = env_info["n_agents"]
    n_enemy = env.env.death_tracker_enemy.shape[0]
    episode_limit = env_info["episode_limit"]
    env.close()
    # close env instance

    env_config = {
        "obs_shape": obs_shape,
        "n_ally": n_ally,
        "n_enemy": n_enemy,
        "state_shape": state_shape,
        "n_actions": n_actions,
        "episode_limit": episode_limit,
    }

    register_env("smac", lambda config: SMAC(args.map))

    ##############
    ### policy ###
    ##############

    if args.share_policy:
        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")
    else:
        policies = {
            "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(n_ally)
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[int(agent_id[6:])])

    policy_function_dict = {
        "PG": run_pg_a2c_a3c,
        "A2C": run_pg_a2c_a3c,
        "A3C": run_pg_a2c_a3c,
        "R2D2": run_r2d2,
        "VDN": run_vdn_qmix_iql,
        "QMIX": run_vdn_qmix_iql,
        "IQL": run_vdn_qmix_iql,
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

    #####################
    ### common config ###
    #####################

    common_config = {
        "seed": random.randint(0, 9999),
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_gpus_per_worker": args.num_gpus_per_worker,
        "env_config": {
            "map_name": args.map,
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "callbacks": SmacCallbacks,
        "logger_config": {
            "type": SMACLogger,
            "prefix": "",
        },
        "framework": args.framework,
        "evaluation_interval": args.evaluation_interval,
    }

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    ################
    ### reporter ###
    ################

    default_metric_columns = SMACReporter.DEFAULT_COLUMNS.copy()
    customized_metric_columns = {
        "custom_metrics/battle_win_rate_mean": "battle_win_rate",
        "custom_metrics/ally_survive_rate_mean": "ally_survive_rate",
        "custom_metrics/enemy_kill_rate_mean": "enemy_kill_rate",
    }
    reporter = SMACReporter(
        metric_columns=merge_dicts(default_metric_columns, customized_metric_columns),
        sort_by_metric=True,
    )

    ##################
    ### run script ###
    ##################

    results = policy_function_dict[args.run](args, common_config, env_config, stop, reporter)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
