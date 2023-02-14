import ray
from gym.spaces import Dict as GymDict, Discrete, Box, Tuple
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marllib.marl.models.zoo.jointQ_rnn import JointQ_RNN
from marllib.marl.models.zoo.vd_rnn import VD_RNN
from marllib.marl.models.zoo.ddpg_rnn import DDPG_RNN
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY as ENV_REGISTRY
from marllib.marl.common import _get_model_config, recursive_dict_update, merge_default_and_customer
from tabulate import tabulate

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def render_vd(config_dict, customer_stop=None):
    ray.init(local_mode=config_dict["local_mode"])

    ###################
    ### environment ###
    ###################

    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/global_reward_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/global_reward_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == config_dict["env"]:
                check_current_used_env_flag = True

    print(tabulate(env_reg_ls,
                   headers=['Env_Name', 'Check_Status', "Error_Log", "Config_File_Location", "Env_File_Location"],
                   tablefmt='grid'))

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                config_dict["env"]))

    map_name = config_dict["env_args"]["map_name"]
    test_env = ENV_REGISTRY[config_dict["env"]](config_dict["env_args"])
    agent_name_ls = test_env.agents
    env_info_dict = test_env.get_env_info()
    test_env.close()

    # grab the policy mapping info here to use in grouping environment
    policy_mapping_info = env_info_dict["policy_mapping_info"]

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if config_dict["algorithm"] in ["qmix", "vdn", "iql"]:
        space_obs = env_info_dict["space_obs"].spaces
        space_act = env_info_dict["space_act"]
        # check the action space condition:
        if not isinstance(space_act, Discrete):
            raise ValueError("illegal action space")

        n_agents = env_info_dict["num_agents"]

        if config_dict["share_policy"] == "all":
            obs_space = Tuple([GymDict(space_obs)] * n_agents)
            act_space = Tuple([space_act] * n_agents)
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError("in {}, policy can not be shared".format(map_name))
            grouping = {"group_all_": agent_name_ls}

        elif config_dict["share_policy"] == "group":
            groups = policy_mapping_info["team_prefix"]
            if len(groups) == 1:
                obs_space = Tuple([GymDict(space_obs)] * n_agents)
                act_space = Tuple([space_act] * n_agents)
                if not policy_mapping_info["all_agents_one_policy"]:
                    raise ValueError("in {}, policy can not be shared".format(map_name))
                grouping = {"group_all_": agent_name_ls}
            else:
                raise ValueError("joint Q learning does not support group function")
        elif config_dict["share_policy"] == "individual":
            raise ValueError("joint Q learning does not support individual function")
        else:
            raise ValueError("wrong share_policy {}".format(config_dict["share_policy"]))

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

    obs_dim = len(env_info_dict["space_obs"]["obs"].shape)

    if obs_dim == 1:
        print("use fc encoder")
        encoder = "fc_encoder"
    else:
        print("use cnn encoder")
        encoder = "cnn_encoder"

    # load model config according to env_info:
    # encoder config
    encoder_arch_config = _get_model_config(encoder)
    config_dict = recursive_dict_update(config_dict, encoder_arch_config)

    # core rnn config
    rnn_arch_config = _get_model_config("rnn")
    config_dict = recursive_dict_update(config_dict, rnn_arch_config)

    # core rnn config
    mixer_arch_config = _get_model_config("mixer")
    config_dict = recursive_dict_update(config_dict, mixer_arch_config)

    ModelCatalog.register_custom_model(
        "Joint_Q_Model", JointQ_RNN)

    ModelCatalog.register_custom_model(
        "Value_Decomposition_Model", VD_RNN)

    ModelCatalog.register_custom_model(
        "DDPG_Model", DDPG_RNN)

    ##############
    ### policy ###
    ##############

    if config_dict["algorithm"] in ["qmix", "vdn", "iql"]:  # the policy mapping is done in script part
        policies = None
        policy_mapping_fn = None

    else:
        if config_dict["share_policy"] == "all":
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError("in {}, policy can not be shared".format(map_name))

            policies = {"shared_policy"}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: "shared_policy")

        elif config_dict["share_policy"] == "group":
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
        "env": env_reg_name,
        "num_gpus_per_worker": config_dict["num_gpus_per_worker"],
        "num_gpus": config_dict["num_gpus"],
        "num_workers": config_dict["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": config_dict["framework"],
        "simple_optimizer": False,  # force using better optimizer
        # Evaluate once per training iteration.
        "evaluation_interval": 1,
        # Run evaluation on (at least) two episodes
        "evaluation_num_episodes": 100,
        # ... using one evaluation worker (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Store videos in this relative directory here inside
            # the default output dir (~/ray_results/...).
            # Alternatively, you can specify an absolute path.
            # Set to True for using the default output dir (~/ray_results/...).
            # Set to False for not recording anything.
            "record_env": False,
            # "record_env": "/Users/xyz/my_videos/",

            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": True,
        },
    }

    stop = {
        "episode_reward_mean": config_dict["stop_reward"],
        "timesteps_total": config_dict["stop_timesteps"],
        "training_iteration": config_dict["stop_iters"],
    }

    stop = merge_default_and_customer(stop, customer_stop)

    ##################
    ### run script ###
    ###################

    results = POlICY_REGISTRY[config_dict["algorithm"]](config_dict, common_config, env_info_dict, stop)

    ray.shutdown()
