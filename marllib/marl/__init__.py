from marllib.marl.common import *
from marllib.marl.algos import run_il, run_vd, run_cc
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.models import *

from ray.tune import register_env
import yaml
import os
import sys
from copy import deepcopy
from tabulate import tabulate

with open(os.path.join(os.path.dirname(__file__), "ray/ray.yaml"), "r") as f:
    CONFIG_DICT = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

SYSPARAMs = deepcopy(sys.argv)


class Env(dict):
    def set_ray(self):

        # default config
        with open(os.path.join(os.path.dirname(__file__), "ray/ray.yaml"), "r") as f:
            ray_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        # user config
        user_ray_args = {}
        for param in SYSPARAMs:
            if param.startswith("--ray_args"):
                if "=" in param:
                    key, value = param.split(".")[1].split("=")
                    user_ray_args[key] = value
                else:  # local_mode
                    user_ray_args[param.split(".")[1]] = True

        # update config
        ray_config_dict = merge_default_and_customized_and_check(ray_config_dict, user_ray_args)

        for key, value in ray_config_dict.items():
            self[key] = value

        return self


def make_env(environment_name,
             map_name,
             force_coop=False,
             **env_params
             ):
    """
    Gets the environment configuration, which will be given to Ray RLlib.
    """

    # default config
    env_config_file_path = os.path.join(os.path.dirname(__file__),
                                        "../envs/base_env/config/{}.yaml".format(environment_name))
    if not os.path.exists(env_config_file_path):
        env_config_file_path = os.path.join(os.path.dirname(__file__),
                                            "../../examples/config/env_config/{}.yaml".format(environment_name))

    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # update function-fixed config
    env_config_dict["env_args"] = merge_default_and_customized_and_check(env_config_dict["env_args"], env_params)

    # user commandline config
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = merge_default_and_customized_and_check(env_config_dict["env_args"], user_env_args)

    env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop
    env_config = Env(env_config_dict)

    # set ray config
    env_config = env_config.set_ray()

    # initialize env
    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == env_config["env"]:
                check_current_used_env_flag = True

    print(tabulate(env_reg_ls,
                   headers=['Env_Name', 'Check_Status', "Error_Log", "Config_File_Location", "Env_File_Location"],
                   tablefmt='grid'))

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                env_config["env"]))

    env_reg_name = env_config["env"] + "_" + env_config["env_args"]["map_name"]

    if env_config["force_coop"]:
        register_env(env_reg_name, lambda _: COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = COOP_ENV_REGISTRY[env_config["env"]](env_config["env_args"])
    else:
        register_env(env_reg_name, lambda _: ENV_REGISTRY[env_config["env"]](env_config["env_args"]))
        env = ENV_REGISTRY[env_config["env"]](env_config["env_args"])

    return env, env_config


def build_model(environment, algorithm, model_preference):
    if algorithm.name in ["ddpg", "facmac", "maddpg"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = DDPG_RNN
        else:
            model_class = DDPG_MLP

    elif algorithm.name in ["qmix", "vdn", "iql"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = JointQ_RNN
        else:
            model_class = JointQ_MLP

    else:
        if algorithm.algo_type == "IL":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = Base_RNN
            else:
                model_class = Base_MLP
        elif algorithm.algo_type == "CC":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = CC_RNN
            else:
                model_class = CC_MLP
        else:  # VD
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = VD_RNN
            else:
                model_class = VD_MLP

    if model_preference["core_arch"] in ["gru", "lstm"]:
        model_config = get_model_config("rnn")
    elif model_preference["core_arch"] in ["mlp"]:
        model_config = get_model_config("mlp")
    else:
        raise NotImplementedError("{} not supported agent model arch".format(model_preference["core_arch"]))

    if len(environment[0].observation_space.spaces["obs"].shape) == 1:
        print("use fc encoder")
        encoder = "fc_encoder"
    else:
        print("use cnn encoder")
        encoder = "cnn_encoder"

    # encoder config
    encoder_arch_config = get_model_config(encoder)
    model_config = recursive_dict_update(model_config, encoder_arch_config)
    model_config = recursive_dict_update(model_config, {"model_arch_args": model_preference})

    if algorithm.algo_type == "VD":
        mixer_arch_config = get_model_config("mixer")
        model_config = recursive_dict_update(model_config, mixer_arch_config)

    return model_class, model_config


class _Algo:
    def __init__(self, algo_name):

        if "_" in algo_name:
            self.name = algo_name.split("_")[0].lower()
            self.algo_type = algo_name.split("_")[1].upper()
        else:
            self.name = algo_name
            self.algo_type = check_algo_type(self.name.lower())
        self.algo_parameters = {}
        self.config_dict = None
        self.common_config = None


    def __call__(self, hyperparam_source='common', **algo_params):
        """
        @param: hyperparam_source:
        1. 'common'             use marl/algos/hyperparams/common
        2. $environment_name    use marl/algos/hyperparams/finetuned/$environment_name
        3. 'test'               use marl/algos/hyperparams/test
        """
        # if '_lambda' in algo_parameters:
        #     algo_parameters['lambda'] = algo_parameters['_lambda']
        #     del algo_parameters['_lambda']
        if hyperparam_source in ["common", "test"]:
            rel_path = "algos/hyperparams/{}/{}.yaml".format(hyperparam_source, self.name)
        else:
            rel_path = "algos/hyperparams/finetuned/{}/{}.yaml".format(hyperparam_source, self.name)

        if not os.path.exists(os.path.join(os.path.dirname(__file__), rel_path)):
            rel_path = "../../examples/config/algo_config/{}.yaml".format(self.name)

        with open(os.path.join(os.path.dirname(__file__), rel_path), "r") as f:
            algo_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        # update function-fixed config
        algo_config_dict['algo_args'] = merge_default_and_customized_and_check(algo_config_dict['algo_args'],
                                                                               algo_params)

        # user config
        user_algo_args = {}
        for param in SYSPARAMs:
            if param.startswith("--algo_args"):
                value = param.split("=")[1]
                key = param.split("=")[0].split(".")[1]
                user_algo_args[key] = value

        # update commandline config
        algo_config_dict['algo_args'] = merge_default_and_customized_and_check(algo_config_dict['algo_args'],
                                                                               user_algo_args)

        self.algo_parameters = algo_config_dict

        return self

    def fit(self, env, model, stop=None, **running_params):

        env_instance, info = env
        model_class, model_info = model

        self.config_dict = info
        self.config_dict = recursive_dict_update(self.config_dict, model_info)

        self.config_dict = recursive_dict_update(self.config_dict, self.algo_parameters)
        self.config_dict = recursive_dict_update(self.config_dict, running_params)

        self.config_dict['algorithm'] = self.name

        if self.algo_type == "IL":
            run_il(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "VD":
            run_vd(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "CC":
            run_cc(self.config_dict, env_instance, model_class, stop=stop)
        else:
            raise ValueError("not supported type {}".format(self.algo_type))

    def render(self, env, model, stop=None, **running_params):
        # current reuse the fit function
        self.fit(env, model, stop, **running_params)


class _AlgoManager:
    def __init__(self):
        # set each algorithm to AlgoManager.
        # could get :
        # happo = marlib.algos.HAPPO()
        for algo_name in POlICY_REGISTRY:
            setattr(_AlgoManager, algo_name, _Algo(algo_name))
            # set set algos.HAPPO = _Algo(run_happo)

    def register_algo(self, algo_name, style, script):
        setattr(_AlgoManager, algo_name, _Algo(algo_name + "_" + style))
        POlICY_REGISTRY[algo_name] = script


algos = _AlgoManager()
