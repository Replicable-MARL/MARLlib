"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray import tune
from ray.tune.registry import register_env
from ray.rllib.utils.test_utils import check_learning_achieved

from MaMujoco.config_mamujoco import get_train_parser
from MaMujoco.env.mamujoco_rllib import RllibMAMujoco
from MaMujoco.util.maddpg_tools import *
from MaMujoco.model.torch_gru import Torch_GRU_Model
from MaMujoco.model.torch_lstm import Torch_LSTM_Model
from MaMujoco.model.torch_gru_cc import Torch_GRU_CentralizedCritic_Model
from MaMujoco.model.torch_lstm_cc import Torch_LSTM_CentralizedCritic_Model
from MaMujoco.model.torch_vd_ppo_a2c_gru_lstm import Torch_LSTM_Model_w_Mixer, Torch_GRU_Model_w_Mixer
from MaMujoco.policy.pg_a2c_a3c import run_pg_a2c_a3c
from MaMujoco.policy.ppo import run_ppo
from MaMujoco.policy.vda2c import run_vda2c_sum_mix
from MaMujoco.policy.vdppo import run_vdppo_sum_mix
from MaMujoco.policy.maa2c import run_maa2c
from MaMujoco.policy.mappo import run_mappo
from MaMujoco.policy.ddpg import run_ddpg
from MaMujoco.policy.maddpg import run_maddpg
from MaMujoco.policy.happo import run_happo
from MaMujoco.policy.hatrpo import run_hatrpo

# from https://github.com/schroederdewitt/multiagent_mujoco

env_args_dict = {
    "2AgentAnt": {"scenario": "Ant-v2",
                  "agent_conf": "2x4",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "2AgentAntDiag": {"scenario": "Ant-v2",
                      "agent_conf": "2x4d",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "4AgentAnt": {"scenario": "Ant-v2",
                  "agent_conf": "4x2",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "2AgentHalfCheetah": {"scenario": "HalfCheetah-v2",
                          "agent_conf": "2x3",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "6AgentHalfCheetah": {"scenario": "HalfCheetah-v2",
                          "agent_conf": "6x1",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "3AgentHopper": {"scenario": "Hopper-v2",
                     "agent_conf": "3x1",
                     "agent_obsk": 0,
                     "episode_limit": 1000},
    "2AgentHumanoid": {"scenario": "Humanoid-v2",
                       "agent_conf": "9|8",
                       "agent_obsk": 1,
                       "episode_limit": 1000},
    "2AgentHumanoidStandup": {"scenario": "HumanoidStandup-v2",
                              "agent_conf": "9|8",
                              "agent_obsk": 1,
                              "episode_limit": 1000},
    "2AgentReacher": {"scenario": "Reacher-v2",
                      "agent_conf": "2x1",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "2AgentSwimmer": {"scenario": "Swimmer-v2",
                      "agent_conf": "2x1",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "2AgentWalker": {"scenario": "Walker2d-v2",
                     "agent_conf": "2x3",
                     "agent_obsk": 1,
                     "episode_limit": 1000},
    "ManyagentSwimmer": {"scenario": "manyagent_swimmer",
                         "agent_conf": "10x2",
                         "agent_obsk": 1,
                         "episode_limit": 1000},
    "ManyagentAnt": {"scenario": "manyagent_ant",
                     "agent_conf": "2x3",
                     "agent_obsk": 1,
                     "episode_limit": 1000},
}

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=False)

    ###################
    ### environment ###
    ###################

    env_config = env_args_dict[args.map]

    register_env(args.map, lambda _: RllibMAMujoco(env_config))

    single_env = RllibMAMujoco(env_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    state_dim = single_env.state_dim
    ally_num = single_env.num_agents

    env_config["state_dim"] = state_dim
    env_config["ally_num"] = ally_num

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
        "PG": run_pg_a2c_a3c,
        "A2C": run_pg_a2c_a3c,
        "A3C": run_pg_a2c_a3c,
        "PPO": run_ppo,
        "MIX-VDA2C": run_vda2c_sum_mix,
        "SUM-VDA2C": run_vda2c_sum_mix,
        "MIX-VDPPO": run_vdppo_sum_mix,
        "SUM-VDPPO": run_vdppo_sum_mix,
        "MAA2C": run_maa2c,
        "MAPPO": run_mappo,
        "DDPG": run_ddpg,
        "MADDPG": run_maddpg,
        "HAPPO": run_happo,
        "HATRPO": run_hatrpo,
    }

    #############
    ### model ###
    #############

    # Independent
    ModelCatalog.register_custom_model("LSTM", Torch_LSTM_Model)
    ModelCatalog.register_custom_model("GRU", Torch_GRU_Model)

    # CTDE(centralized critic (only action))
    ModelCatalog.register_custom_model(
        "GRU_CentralizedCritic", Torch_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "LSTM_CentralizedCritic", Torch_LSTM_CentralizedCritic_Model)

    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("GRU_ValueMixer", Torch_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("LSTM_ValueMixer", Torch_LSTM_Model_w_Mixer)
    # ModelCatalog.register_custom_model("CNN_UPDeT_ValueMixer", Torch_CNN_Transformer_Model_w_Mixer)

    #####################
    ### common config ###
    #####################

    common_config = {
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

    policy = policy_function_dict[args.run]

    results = policy(args, common_config, env_config, stop)

    if args.test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
