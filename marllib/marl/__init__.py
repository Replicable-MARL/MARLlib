# --algo_config=happo
# --finetuned
# --env-config=mamujoco with env_args.map_name=2AgentWalker

from marllib.marl.algos.scripts import POlICY_REGISTRY
from marllib.marl.common import _get_config, recursive_dict_update, check_algo_type
from marllib.envs.base_env import ENV_REGISTRY
from ray.tune import register_env
from marllib.marl.algos.run_il import run_il
from marllib.marl.algos.run_vd import run_vd
from marllib.marl.algos.run_cc import run_cc
from marllib.marl.render.render_cc import render_cc
from marllib.marl.render.render_il import render_il
from marllib.marl.render.render_vd import render_vd

from marllib.marl.common import merge_default_and_customer_and_check
import yaml
import os
import sys
from copy import deepcopy

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
        ray_config_dict = merge_default_and_customer_and_check(ray_config_dict, user_ray_args)

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
    with open(os.path.join(os.path.dirname(__file__), "../envs/base_env/config/{}.yaml".format(environment_name)),
              "r") as f:
        env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    # update function-fixed config
    env_config_dict["env_args"] = merge_default_and_customer_and_check(env_config_dict["env_args"], env_params)

    # user commandline config
    user_env_args = {}
    for param in SYSPARAMs:
        if param.startswith("--env_args"):
            key, value = param.split(".")[1].split("=")
            user_env_args[key] = value

    # update commandline config
    env_config_dict["env_args"] = merge_default_and_customer_and_check(env_config_dict["env_args"], user_env_args)

    env_config_dict["env_args"]["map_name"] = map_name
    env_config_dict["force_coop"] = force_coop
    env_config = Env(env_config_dict)

    # set ray config
    env_config = env_config.set_ray()

    return env_config


class _Algo:
    def __init__(self, algo_name):
        self.name = algo_name
        self.algo_parameters = {}
        self.config_dict = None
        self.common_config = None
        self.algo_type = check_algo_type(self.name.lower())

    def __call__(self, hyperparam_source='common', **algo_params):
        """
        @param: hyperparam_source:
        1. 'common'             use marl/algos/hyperparams/common
        2. $environment_name    use marl/algos/hyperparams/finetuned/$environment_name
        """
        # if '_lambda' in algo_parameters:
        #     algo_parameters['lambda'] = algo_parameters['_lambda']
        #     del algo_parameters['_lambda']
        if hyperparam_source == 'common':
            rel_path = "algos/hyperparams/common/{}.yaml".format(self.name)
        else:
            rel_path = "algos/hyperparams/finetuned/{}/{}.yaml".format(hyperparam_source, self.name)

        # default config
        with open(os.path.join(os.path.dirname(__file__), rel_path), "r") as f:
            algo_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        # update function-fixed config
        algo_config_dict['algo_args'] = merge_default_and_customer_and_check(algo_config_dict['algo_args'],
                                                                             algo_params)

        # user config
        user_algo_args = {}
        for param in SYSPARAMs:
            if param.startswith("--algo_args"):
                value = param.split("=")[1]
                key = param.split("=")[0].split(".")[1]
                user_algo_args[key] = value

        # update commandline config
        algo_config_dict['algo_args'] = merge_default_and_customer_and_check(algo_config_dict['algo_args'],
                                                                             user_algo_args)

        self.algo_parameters = algo_config_dict

        return self

    def fit(self, env_config_dict, stop=None, **running_params):
        # env_config, env_dict = env
        # self.common_config['env'] = env_config

        # test_env = ENV_REGISTRY[env_config_dict["env"]](env_config_dict["env_args"])
        # env_info_dict = test_env.get_env_info()

        # test_env.close()

        # need split to IL, CC, VD ...

        self.config_dict = recursive_dict_update(CONFIG_DICT, env_config_dict)
        self.config_dict = recursive_dict_update(self.config_dict, self.algo_parameters)
        self.config_dict = recursive_dict_update(self.config_dict, running_params)

        self.config_dict['algorithm'] = self.name

        if self.algo_type == "IL":
            run_il(self.config_dict, customer_stop=stop)
        elif self.algo_type == "VD":
            run_vd(self.config_dict, customer_stop=stop)
        elif self.algo_type == "CC":
            run_cc(self.config_dict, customer_stop=stop)
        else:
            raise ValueError("algo not in supported algo_type")

    def render(self, env_config_dict, stop=None, **running_params):
        # env_config, env_dict = env
        # self.common_config['env'] = env_config

        # test_env = ENV_REGISTRY[env_config_dict["env"]](env_config_dict["env_args"])
        # env_info_dict = test_env.get_env_info()

        # test_env.close()

        # need split to IL, CC, VD ...

        self.config_dict = recursive_dict_update(CONFIG_DICT, env_config_dict)
        self.config_dict = recursive_dict_update(self.config_dict, self.algo_parameters)
        self.config_dict = recursive_dict_update(self.config_dict, running_params)

        self.config_dict['algorithm'] = self.name

        if self.algo_type == "IL":
            render_il(self.config_dict, customer_stop=stop)
        elif self.algo_type == "VD":
            render_vd(self.config_dict, customer_stop=stop)
        elif self.algo_type == "CC":
            render_cc(self.config_dict, customer_stop=stop)
        else:
            raise ValueError("algo not in supported algo_type")


class _AlgoManager:
    def __init__(self):
        # set each algorithm to AlgoManager.
        # could get :
        # happo = marlib.algos.HAPPO()
        for algo_name in POlICY_REGISTRY:
            setattr(_AlgoManager, algo_name, _Algo(algo_name))
            # set set algos.HAPPO = _Algo(run_happo)


algos = _AlgoManager()
