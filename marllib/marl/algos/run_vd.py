# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ray
import gym
from gym.spaces import Dict as GymDict, Discrete, Tuple
from ray.tune import register_env
from ray import tune
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY as ENV_REGISTRY
from marllib.marl.common import recursive_dict_update, dict_update
from marllib.marl.algos.run_cc import restore_config_update

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def run_vd(exp_info, env, model, stop=None):
    ray.init(local_mode=exp_info["local_mode"], num_gpus=exp_info["num_gpus"])

    ########################
    ### environment info ###
    ########################

    env_info = env.get_env_info()
    map_name = exp_info['env_args']['map_name']
    agent_name_ls = env.agents
    env_info["agent_name_ls"] = agent_name_ls
    env.close()

    ###################
    ### space check ###
    ###################

    action_discrete = isinstance(env_info["space_act"], gym.spaces.Discrete)
    action_multi_discrete = isinstance(env_info["space_act"], gym.spaces.MultiDiscrete)

    if action_discrete or action_multi_discrete:
        if exp_info["algorithm"] in ["facmac"]:
            raise ValueError(
                "Algo -facmac- only supports continuous action space,  Env {} requires Discrete action space".format(
                    exp_info["env"]))
        if action_multi_discrete:
            if exp_info["algorithm"] in ["vdn", "qmix"]:
                raise ValueError(
                    "Algo -{}- only supports discrete action space,  Env -{}- requires MultiDiscrete action space".format(
                        env_info["algorithm"],
                        exp_info["env"]))
    else:  # continuous
        if exp_info["algorithm"] in ["coma", "vdn", "qmix"]:
            raise ValueError(
                "Algo -{}- only supports discrete action space, Env -{}- requires continuous action space".format(
                    env_info["algorithm"],
                    exp_info["env"]))

    ######################
    ### policy sharing ###
    ######################

    # grab the policy mapping info here to use in grouping environment
    policy_mapping_info = env_info["policy_mapping_info"]

    shared_policy_name = "default_policy" if exp_info["agent_level_batch_update"] else "shared_policy"
    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if exp_info["algorithm"] in ["qmix", "vdn", "iql"]:
        space_obs = env_info["space_obs"].spaces
        space_act = env_info["space_act"]
        # check the action space condition:
        if not isinstance(space_act, Discrete):
            raise ValueError("illegal action space")

        n_agents = env_info["num_agents"]

        if exp_info["share_policy"] == "all":
            obs_space = Tuple([GymDict(space_obs)] * n_agents)
            act_space = Tuple([space_act] * n_agents)
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError("in {}, policy can not be shared".format(map_name))
            grouping = {"group_all_": agent_name_ls}

        elif exp_info["share_policy"] == "group":
            groups = policy_mapping_info["team_prefix"]
            if len(groups) == 1:
                obs_space = Tuple([GymDict(space_obs)] * n_agents)
                act_space = Tuple([space_act] * n_agents)
                if not policy_mapping_info["all_agents_one_policy"]:
                    raise ValueError("in {}, policy can not be shared".format(map_name))
                grouping = {"group_all_": agent_name_ls}
            else:
                raise ValueError("joint Q learning does not support group function")
        elif exp_info["share_policy"] == "individual":
            raise ValueError("joint Q learning does not support individual function")
        else:
            raise ValueError("wrong share_policy {}".format(exp_info["share_policy"]))

        env_reg_name = "grouped_" + exp_info["env"] + "_" + exp_info["env_args"]["map_name"]
        register_env(env_reg_name,
                     lambda _: ENV_REGISTRY[exp_info["env"]](exp_info["env_args"]).with_agent_groups(
                         grouping, obs_space=obs_space, act_space=act_space))
    else:
        env_reg_name = exp_info["env"] + "_" + exp_info["env_args"]["map_name"]
        register_env(env_reg_name,
                     lambda _: ENV_REGISTRY[exp_info["env"]](exp_info["env_args"]))

    if exp_info["algorithm"] in ["qmix", "vdn", "iql"]:
        policies = None
        policy_mapping_fn = None

    else:
        if exp_info["share_policy"] == "all":
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError("in {}, policy can not be shared".format(map_name))

            policies = {shared_policy_name}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: shared_policy_name)

        elif exp_info["share_policy"] == "group":
            groups = policy_mapping_info["team_prefix"]
            if len(groups) == 1:
                if not policy_mapping_info["all_agents_one_policy"]:
                    raise ValueError(
                        "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))
                policies = {shared_policy_name}
                policy_mapping_fn = (
                    lambda agent_id, episode, **kwargs: shared_policy_name)
            else:
                policies = {
                    "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
                    groups
                }
                policy_ids = list(policies.keys())
                policy_mapping_fn = tune.function(
                    lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

        elif exp_info["share_policy"] == "individual":
            if not policy_mapping_info["one_agent_one_policy"]:
                raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

            policies = {
                "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
                range(env_info["num_agents"])
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

        else:
            raise ValueError("wrong share_policy {}".format(exp_info["share_policy"]))

    #########################
    ### experiment config ###
    #########################

    run_config = {
        "seed": int(exp_info["seed"]),
        "env": env_reg_name,
        "num_gpus_per_worker": exp_info["num_gpus_per_worker"],
        "num_gpus": exp_info["num_gpus"],
        "num_workers": exp_info["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": exp_info["framework"],
        "evaluation_interval": exp_info["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop_config = {
        "episode_reward_mean": exp_info["stop_reward"],
        "timesteps_total": exp_info["stop_timesteps"],
        "training_iteration": exp_info["stop_iters"],
    }

    stop_config = dict_update(stop_config, stop)

    exp_info, run_config, stop_config, restore_config = restore_config_update(exp_info, run_config, stop_config)

    ##################
    ### run script ###
    ###################

    results = POlICY_REGISTRY[exp_info["algorithm"]](model, exp_info, run_config, env_info, stop_config,
                                                     restore_config)
    ray.shutdown()

    return results

