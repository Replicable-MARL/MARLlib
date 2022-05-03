import yaml
import os
import sys
import collections
import ray
from copy import deepcopy
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marl.models.IL.onpolicy_rnn import Onpolicy_Universal_Model
from envs.base_env import ENV_REGISTRY
from marl.algos.scripts import POlICY_REGISTRY
from marl.common import _get_model_config, recursive_dict_update

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def run_il(config_dict):

    ray.init(local_mode=config_dict["local_mode"])

    ###################
    ### environment ###
    ###################

    register_env(config_dict["env"] + "_" + config_dict["env_args"]["map_name"],
                 lambda _: ENV_REGISTRY[config_dict["env"]](config_dict["env_args"]))

    map_name = config_dict["env_args"]["map_name"]
    test_env = ENV_REGISTRY[config_dict["env"]](config_dict["env_args"])
    env_info_dict = test_env.get_env_info()
    agent_name_ls = test_env.agents
    test_env.close()

    #############
    ### model ###
    #############

    obs_dim = len(env_info_dict["space_obs"]["obs"].shape)

    if obs_dim == 1:
        print("use fc encoder")
        encoder = "fc_encoder"
    else:
        print("use cnn encoder")
        encoder = "cnn_encoder"

    # load model config according to env_info:
    # encoder config
    encoder_arch_config = _get_model_config({}, encoder)
    config_dict = recursive_dict_update(config_dict, encoder_arch_config)

    # core rnn config
    rnn_arch_config = _get_model_config({}, "rnn")
    config_dict = recursive_dict_update(config_dict, rnn_arch_config)

    ModelCatalog.register_custom_model(
        "Universal_Model", Onpolicy_Universal_Model)

    ##############
    ### policy ###
    ##############

    policy_mapping_info = env_info_dict["policy_mapping_info"]

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if config_dict["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared".format(map_name))

        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")

    elif config_dict["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]
        policies = {
            "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
            groups
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif config_dict["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
            range(env_info_dict["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(config_dict["share_policy"]))

    #####################
    ### common config ###
    #####################

    common_config = {
        "seed": config_dict["seed"],
        "env": config_dict["env"] + "_" + config_dict["env_args"]["map_name"],
        "num_gpus_per_worker": config_dict["num_gpus_per_worker"],
        "num_gpus": config_dict["num_gpus"],
        "num_workers": config_dict["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": config_dict["framework"],
        "evaluation_interval": config_dict["evaluation_interval"],
    }

    stop = {
        "episode_reward_mean": config_dict["stop_reward"],
        "timesteps_total": config_dict["stop_timesteps"],
        "training_iteration": config_dict["stop_iters"],
    }

    ##################
    ### run script ###
    ###################

    results = POlICY_REGISTRY[config_dict["algorithm"]](config_dict, common_config, env_info_dict, stop)

    if config_dict.as_test:
        check_learning_achieved(results, config_dict.stop_reward)

    ray.shutdown()
