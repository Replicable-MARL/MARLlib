import ray
from ray import tune
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.marl.common import recursive_dict_update, merge_default_and_customized

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def run_cc(algo_config, env, model, stop=None):
    ray.init(local_mode=algo_config["local_mode"])

    ########################
    ### environment info ###
    ########################

    env_config = env.get_env_info()
    map_name = algo_config['env_args']['map_name']
    agent_name_ls = env.agents
    env_config["agent_name_ls"] = agent_name_ls
    env.close()

    ######################
    ### policy sharing ###
    ######################

    policy_mapping_info = env_config["policy_mapping_info"]

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if algo_config["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError("in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

        policies = {"av"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "av")

    elif algo_config["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

            policies = {"shared_policy"}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: "shared_policy")

        else:
            policies = {
                "policy_{}".format(i): (None, env_config["space_obs"], env_config["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif algo_config["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_config["space_obs"], env_config["space_act"], {}) for i in
            range(env_config["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(algo_config["share_policy"]))

    # if happo or hatrpo, force individual
    if algo_config["algorithm"] in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError("in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_config["space_obs"], env_config["space_act"], {}) for i in
            range(env_config["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    #########################
    ### experiment config ###
    #########################

    common_config = {
        "seed": int(algo_config["seed"]),
        "env": algo_config["env"] + "_" + algo_config["env_args"]["map_name"],
        "num_gpus_per_worker": algo_config["num_gpus_per_worker"],
        "num_gpus": algo_config["num_gpus"],
        "num_workers": algo_config["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": algo_config["framework"],
        "evaluation_interval": algo_config["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop_config = {
        "episode_reward_mean": algo_config["stop_reward"],
        "timesteps_total": algo_config["stop_timesteps"],
        "training_iteration": algo_config["stop_iters"],
    }

    stop_config = merge_default_and_customized(stop_config, stop)

    if algo_config['restore_path']['model_path'] == '':
        restore = None
    else:
        restore = algo_config['restore_path']
        render_config = {
            "evaluation_interval": 1,
            "evaluation_num_episodes": 100,
            "evaluation_num_workers": 1,
            "evaluation_config": {
                "record_env": False,
                "render_env": True,
            }
        }

        common_config = recursive_dict_update(common_config, render_config)

        render_stop_config = {
            "training_iteration": 1,
        }

        stop_config = recursive_dict_update(stop_config, render_stop_config)

    ##################
    ### run script ###
    ##################

    results = POlICY_REGISTRY[algo_config["algorithm"]](model, algo_config, common_config, env_config, stop_config,
                                                        restore)

    ray.shutdown()
