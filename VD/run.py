import yaml
import os
import sys
import collections
import ray
from copy import deepcopy
from gym.spaces import Dict as GymDict, Discrete, Box, Tuple
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from VD.models.offpolicy_rnn import Offpolicy_Universal_Model
from VD.models.onpolicy_rnn import Onpolicy_Universal_Model
from VD.scripts import POlICY_REGISTRY
from VD.envs import ENV_REGISTRY

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def _get_config(params, arg_name, subfolder):
    config_name = None

    if params == {}:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(arg_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(arg_name, exc)
        return config_dict

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == "__main__":

    ######################
    ### Prepare Config ###
    ######################

    params = deepcopy(sys.argv)
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config/base", "ray_base.yaml"), "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # env config
    env_config = _get_config(params, "--env-config", "envs")
    config_dict = recursive_dict_update(config_dict, env_config)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
            config_dict["env_args"]["map_name"] = map_name

    # algorithms config
    for param in params:
        if param.startswith("--algo_config"):
            algo_name = param.split("=")[1]
            config_dict["algorithm"] = algo_name

    algo_config = _get_config(params, "--algo_config", "algos")
    config_dict = recursive_dict_update(config_dict, algo_config)

    ray.init(local_mode=config_dict["local_mode"])

    ###################
    ### environment ###
    ###################

    test_env = ENV_REGISTRY[config_dict["env"]](config_dict["env_args"])
    env_info_dict = test_env.get_env_info()
    test_env.close()

    if config_dict["algorithm"] in ["qmix", "vdn", "iql"]:

        space_obs = env_info_dict["space_obs"].spaces
        space_act = env_info_dict["space_act"]
        # check the action space condition:
        if not isinstance(space_act, Discrete):
            print("illegal action space")
            raise ValueError()

        n_agents = env_info_dict["num_agents"]
        obs_space = Tuple([GymDict(space_obs)] * n_agents)
        act_space = Tuple([space_act] * n_agents)

        grouping = {
            "group_1": ["agent_{}".format(i) for i in range(n_agents)],
        }
        env_reg_name = "grouped_" + config_dict["env"] + "_" + config_dict["env_args"]["map_name"]
        register_env(env_reg_name,
                     lambda _: ENV_REGISTRY[config_dict["env"]](config_dict["env_args"]).with_agent_groups(
                         grouping, obs_space=obs_space, act_space=act_space))
    else:
        env_reg_name = config_dict["env"] + "_" + config_dict["env_args"]["map_name"]
        register_env(env_reg_name,
                     lambda _: ENV_REGISTRY[config_dict["env"]](config_dict["env_args"]))


    #############
    ### model ###
    #############
    if isinstance(env_info_dict["space_obs"], GymDict):
        obs_dim = len(env_info_dict["space_obs"]["obs"].shape)

    elif isinstance(env_info_dict["space_obs"], Box):
        obs_dim = len(env_info_dict["space_obs"].shape)

    if obs_dim == 1:
        print("use fc encoder")
        encoder = "fc_encoder"
    else:
        print("use cnn encoder")
        encoder = "cnn_encoder"

    # load model config according to env_info:
    # encoder config
    encoder_arch_config = _get_config({}, encoder, "models")
    config_dict = recursive_dict_update(config_dict, encoder_arch_config)

    # core rnn config
    rnn_arch_config = _get_config({}, "rnn", "models")
    config_dict = recursive_dict_update(config_dict, rnn_arch_config)

    # core rnn config
    mixer_arch_config = _get_config({}, "mixer", "models")
    config_dict = recursive_dict_update(config_dict, mixer_arch_config)

    ModelCatalog.register_custom_model(
        "Offpolicy_Universal_Model", Offpolicy_Universal_Model)

    ModelCatalog.register_custom_model(
        "Onpolicy_Universal_Model", Onpolicy_Universal_Model)
    ##############
    ### policy ###
    ##############

    if config_dict["share_policy"]:
        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")
    else:
        policies = {
            "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
            range(env_info_dict["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[int(agent_id[6:])])

    #####################
    ### common config ###
    #####################

    common_config = {
        "seed": config_dict["seed"],
        "env": env_reg_name,
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
