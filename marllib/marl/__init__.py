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

from marllib.marl.common import dict_update, get_model_config, check_algo_type, \
    recursive_dict_update
from marllib.marl.algos import run_il, run_vd, run_cc
from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl.models import BaseRNN, BaseMLP, CentralizedCriticRNN, CentralizedCriticMLP, ValueDecompRNN, \
    ValueDecompMLP, JointQMLP, JointQRNN, DDPGSeriesRNN, DDPGSeriesMLP
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env
from copy import deepcopy
from tabulate import tabulate
from typing import Any, Dict, Tuple
import yaml
import os
import sys

SYSPARAMs = deepcopy(sys.argv)


def set_ray(config: Dict):
    """
    function of combining ray config with other configs
    :param config: dictionary of config to be combined with
    """
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
    ray_config_dict = dict_update(ray_config_dict, user_ray_args, True)

    for key, value in ray_config_dict.items():
        config[key] = value

    return config


def make_env(
        environment_name: str,
        map_name: str,
        force_coop: bool = False,
        abs_path: str = "",
        **env_params
) -> Tuple[MultiAgentEnv, Dict]:
    """
    construct the environment and register.
    Args:
        :param environment_name: name of the environment
        :param map_name: name of the scenario
        :param force_coop: enforce the reward return of the environment to be global
        :param abs_path: env configuration path
        :param env_params: parameters that can be pass to the environment for customizing the environment

    Returns:
        Tuple[MultiAgentEnv, Dict]: env instance & env configuration dict
    """
    if abs_path != "":
        env_config_file_path = os.path.join(os.path.dirname(__file__), abs_path)
    else:
        # default config
        env_config_file_path = os.path.join(os.path.dirname(__file__),
                                            "../envs/base_env/config/{}.yaml".format(environment_name))

    with open(env_config_file_path, "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # update function-fixed config
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], env_params, True)

    # user commandline config
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = dict_update(env_config_dict["env_args"], user_env_args, True)
    env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop

    # combine with exp running config
    env_config = set_ray(env_config_dict)

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


def build_model(
        environment: Tuple[MultiAgentEnv, Dict],
        algorithm: str,
        model_preference: Dict,
) -> Tuple[Any, Dict]:
    """
    construct the model
    Args:
        :param environment: name of the environment
        :param algorithm: name of the algorithm
        :param model_preference:  parameters that can be pass to the model for customizing the model

    Returns:
        Tuple[Any, Dict]: model class & model configuration
    """

    if algorithm.name in ["iddpg", "facmac", "maddpg"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = DDPGSeriesRNN
        else:
            model_class = DDPGSeriesMLP

    elif algorithm.name in ["qmix", "vdn", "iql"]:
        if model_preference["core_arch"] in ["gru", "lstm"]:
            model_class = JointQRNN
        else:
            model_class = JointQMLP

    else:
        if algorithm.algo_type == "IL":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = BaseRNN
            else:
                model_class = BaseMLP
        elif algorithm.algo_type == "CC":
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = CentralizedCriticRNN
            else:
                model_class = CentralizedCriticMLP
        else:  # VD
            if model_preference["core_arch"] in ["gru", "lstm"]:
                model_class = ValueDecompRNN
            else:
                model_class = ValueDecompMLP

    if model_preference["core_arch"] in ["gru", "lstm"]:
        model_config = get_model_config("rnn")
    elif model_preference["core_arch"] in ["mlp"]:
        model_config = get_model_config("mlp")
    else:
        raise NotImplementedError("{} not supported agent model arch".format(model_preference["core_arch"]))

    if len(environment[0].observation_space.spaces["obs"].shape) == 1:
        encoder = "fc_encoder"
    else:
        encoder = "cnn_encoder"

    # encoder config
    encoder_arch_config = get_model_config(encoder)
    model_config = recursive_dict_update(model_config, encoder_arch_config)
    model_config = recursive_dict_update(model_config, {"model_arch_args": model_preference})

    if algorithm.algo_type == "VD":
        mixer_arch_config = get_model_config("mixer")
        model_config = recursive_dict_update(model_config, mixer_arch_config)
        if "mixer_arch" in model_preference:
            recursive_dict_update(model_config, {"model_arch_args": model_preference})

    return model_class, model_config


class _Algo:
    """An algorithm tool class
    :param str algo_name: the algorithm name
    """

    def __init__(self, algo_name: str):

        if "_" in algo_name:
            self.name = algo_name.split("_")[0].lower()
            self.algo_type = algo_name.split("_")[1].upper()
        else:
            self.name = algo_name
            self.algo_type = check_algo_type(self.name.lower())
        self.algo_parameters = {}
        self.config_dict = None
        self.common_config = None

    def __call__(self, hyperparam_source: str, **algo_params):
        """
        Args:
            :param hyperparam_source: source of the algorithm's hyperparameter
            options:
            1. "common" use config under "marl/algos/hyperparams/common"
            2. $environment use config under "marl/algos/hyperparams/finetuned/$environment"
            3. "test" use config under "marl/algos/hyperparams/test"
        Returns:
            _Algo
        """
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
        algo_config_dict["algo_args"] = dict_update(algo_config_dict["algo_args"],
                                                    algo_params, True)

        # user config
        user_algo_args = {}
        for param in SYSPARAMs:
            if param.startswith("--algo_args"):
                value = param.split("=")[1]
                key = param.split("=")[0].split(".")[1]
                user_algo_args[key] = value

        # update commandline config
        algo_config_dict["algo_args"] = dict_update(algo_config_dict["algo_args"],
                                                    user_algo_args, True)

        self.algo_parameters = algo_config_dict

        return self

    def fit(self, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], stop: Dict = None,
            **running_params) -> None:
        """
        Entering point of the whole training
        Args:
            :param env: a tuple of environment instance and environmental configuration
            :param model: a tuple of model class and model configuration
            :param stop: dict of running stop condition
            :param running_params: other configuration to customize the training
        Returns:
            None
        """

        env_instance, info = env
        model_class, model_info = model

        self.config_dict = info
        self.config_dict = recursive_dict_update(self.config_dict, model_info)

        self.config_dict = recursive_dict_update(self.config_dict, self.algo_parameters)
        self.config_dict = recursive_dict_update(self.config_dict, running_params)

        self.config_dict['algorithm'] = self.name

        if self.algo_type == "IL":
            return run_il(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "VD":
            return run_vd(self.config_dict, env_instance, model_class, stop=stop)
        elif self.algo_type == "CC":
            return run_cc(self.config_dict, env_instance, model_class, stop=stop)
        else:
            raise ValueError("not supported type {}".format(self.algo_type))

    def render(self, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], stop: Dict = None,
               **running_params) -> None:
        """
        Entering point of the rendering, running a one iteration fit instead
        Args:
            :param env: a tuple of environment instance and environmental configuration
            :param model: a tuple of model class and model configuration
            :param stop: dict of running stop condition
            :param running_params: other configuration to customize the rendering
        Returns:
            None
        """

        self.fit(env, model, stop, **running_params)


class _AlgoManager:
    def __init__(self):
        """An algorithm pool class
        """
        for algo_name in POlICY_REGISTRY:
            setattr(_AlgoManager, algo_name, _Algo(algo_name))

    def register_algo(self, algo_name: str, style: str, script: Any):
        """
        Algorithm registration
        Args:
            :param algo_name: algorithm name
            :param style: algorithm learning style from ["il", "vd", "cc"]
            :param script: a running script to start training
        Returns:
            None
        """
        setattr(_AlgoManager, algo_name, _Algo(algo_name + "_" + style))
        POlICY_REGISTRY[algo_name] = script


algos = _AlgoManager()
