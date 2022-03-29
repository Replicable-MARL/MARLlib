from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray import tune
from ray.tune.registry import register_env
from Pommerman.config_pommerman import get_train_parser
from Pommerman.env.pommerman_rllib import RllibPommerman
from Pommerman.model.torch_cnn_lstm import Torch_CNN_LSTM_Model
from Pommerman.model.torch_cnn_gru import Torch_CNN_GRU_Model
from Pommerman.model.torch_cnn_gru_cc import Torch_CNN_GRU_CentralizedCritic_Model
from Pommerman.model.torch_cnn_lstm_cc import Torch_CNN_LSTM_CentralizedCritic_Model
from Pommerman.model.torch_cnn import Torch_CNN_Model
from Pommerman.model.torch_vd_ppo_a2c_gru_lstm import Torch_CNN_GRU_Model_w_Mixer, Torch_CNN_LSTM_Model_w_Mixer
from Pommerman.agent.simple_agent import SimpleAgent
from Pommerman.agent.trainable_place_holder_agent import PlaceHolderAgent
from Pommerman.agent.random_agent import RandomAgent
from Pommerman.policy.pg_a2c_a3c_r2d2 import run_pg_a2c_a3c_r2d2
from Pommerman.policy.vdn_qmix import run_vdn_qmix
from Pommerman.policy.ppo import run_ppo
from Pommerman.policy.vda2c import run_vda2c_sum_mix
from Pommerman.policy.vdppo import run_vdppo_sum_mix
from Pommerman.policy.maa2c import run_maa2c
from Pommerman.policy.mappo import run_mappo
from Pommerman.policy.coma import run_coma

from Pommerman.util.vdppo_tools import *

if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    agent_position = args.agent_position

    if "One" in args.map:
        agent_set = {0, 1}
    else:
        agent_set = {0, 1, 2, 3}

    neural_agent_pos = []
    for i in agent_position:
        neural_agent_pos.append(int(i))
        agent_set.remove(int(i))
    rule_agent_pos = list(agent_set)

    # for 4 agents bomber battle, neural_agent_pos/rule_agent_pos can be [a] to [a,b,c,d]
    # for 2 agents bomber battle, neural_agent_pos should only be [0,1], and rule_agent_pos should only be []
    env_config = {
        "map": args.map,
        "neural_agent_pos": neural_agent_pos,
        "rule_agent_pos": rule_agent_pos,
        "rule_agent_type": args.builtin_ai_type  # human_rule random_rule
    }

    agent_number = len(env_config["neural_agent_pos"])

    if "One" in args.map:
        agent_list = [None, None, ]
        if set(env_config["neural_agent_pos"] + env_config["rule_agent_pos"]) != {0, 1}:
            print("Wrong bomber agent position")
            raise ValueError()

    else:
        agent_list = [None, None, None, None]
        if set(env_config["neural_agent_pos"] + env_config["rule_agent_pos"]) != {0, 1, 2, 3}:
            print("Wrong bomber agent position")
            raise ValueError()

    for agent_pos in env_config["neural_agent_pos"]:
        agent_list[agent_pos] = PlaceHolderAgent()  # fake, just for initialization

    for agent_pos in env_config["rule_agent_pos"]:
        if args.builtin_ai_type == "human_rule":
            agent_list[agent_pos] = SimpleAgent()  # Built-in AI for initialization
        elif args.builtin_ai_type == "random_rule":
            agent_list[agent_pos] = RandomAgent()  # Built-in AI for initialization

    register_env("pommerman", lambda _: RllibPommerman(env_config, agent_list))

    single_env = RllibPommerman(env_config, agent_list)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    ##############
    ### policy ###
    ##############

    policies = {
        "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(agent_number)
    }
    policy_ids = list(policies.keys())

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
    ModelCatalog.register_custom_model("CNN", Torch_CNN_Model)
    ModelCatalog.register_custom_model("CNN_LSTM", Torch_CNN_LSTM_Model)
    ModelCatalog.register_custom_model("CNN_GRU", Torch_CNN_GRU_Model)

    # CTDE(centralized critic)
    ModelCatalog.register_custom_model(
        "CNN_GRU_CentralizedCritic", Torch_CNN_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "CNN_LSTM_CentralizedCritic", Torch_CNN_LSTM_CentralizedCritic_Model)

    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("CNN_GRU_ValueMixer", Torch_CNN_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("CNN_LSTM_ValueMixer", Torch_CNN_LSTM_Model_w_Mixer)


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
            "policy_mapping_fn": tune.function(
                lambda agent_id: policy_ids[int(agent_id[6:])]),
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

    results = policy_function_dict[args.run](args, common_config, env_config, agent_list, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
