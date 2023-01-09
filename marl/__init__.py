#--algo_config=happo
# --finetuned
# --env-config=mamujoco with env_args.map_name=2AgentWalker

from marl.algos.scripts import POlICY_REGISTRY
from marl.common import _get_config, recursive_dict_update, check_algo_type
from envs.base_env import ENV_REGISTRY
from ray.tune import register_env
from marl.algos.run_il import run_il
from marl.algos.run_vd import run_vd
from marl.algos.run_cc import run_cc
import os
import yaml


with open(os.path.join(os.path.dirname(__file__), "ray/ray.yaml"), "r") as f:
    CONFIG_DICT = yaml.load(f, Loader=yaml.FullLoader)
    f.close()


class Env(dict):
    def set_ray(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value

        return self


def make_env(environment_name,
             map_name,
             mask_flag,
             global_state_flag,
             opp_action_in_cc,
             fixed_batch_timesteps=None,
             core_arch=None,
             **env_args
             ):
    """
    Gets the environment configuration, which will be given to Ray RLlib.
    """
    env_args['map_name'] = map_name

    env_config_dict = Env({
        'env': environment_name,
        'env_args': env_args,
        'mask_flag': mask_flag,
        'global_state_flag': global_state_flag,
        'opp_action_in_cc': opp_action_in_cc,
        'fixed_batch_timesteps': fixed_batch_timesteps,
        'core_arch': core_arch,
    })

    return env_config_dict


class _Algo:
    def __init__(self, algo_name):
        self.name = algo_name
        self.algo_parameters = {}
        self.config_dict = None
        self.common_config = None
        self.algo_type = check_algo_type(self.name.lower())

    def __call__(self, **algo_parameters):
        """
        @param: config_dict, the configuration of this algorithm
        """
        if '_lambda' in algo_parameters:
            algo_parameters['lambda'] = algo_parameters['_lambda']
            del algo_parameters['_lambda']

        self.algo_parameters['algo_args'] = algo_parameters

    def fit_online(self, env_config_dict, stop=None, **running_parameters):
        # env_config, env_dict = env
        # self.common_config['env'] = env_config

        # test_env = ENV_REGISTRY[env_config_dict["env"]](env_config_dict["env_args"])
        # env_info_dict = test_env.get_env_info()

        # test_env.close()

        # need split to IL, CC, VD ...
        self.config_dict['algorithm'] = self.name

        self.config_dict = recursive_dict_update(CONFIG_DICT, env_config_dict)
        self.config_dict = recursive_dict_update(self.config_dict, self.algo_parameters)

        if self.algo_type == "IL":
            run_il(self.config_dict, customer_config=running_parameters, customer_stop=stop)
        elif self.algo_type == "VD":
            run_vd(self.config_dict, customer_config=running_parameters, customer_stop=stop)
        elif self.algo_type == "CC":
            run_cc(self.config_dict, customer_config=running_parameters, customer_stop=stop)
        else:
            raise ValueError("algo not in supported algo_type")


class _AlgoManager:
    def __init__(self):
        # set each algorithm to AlgoManager.
        # could get :
        # happo = marlib.algos.HAPPO()
        for algo_name in POlICY_REGISTRY:
            setattr(_AlgoManager, algo_name.upper(), _Algo(algo_name))
            # set set algos.HAPPO = _Algo(run_happo)


algos = _AlgoManager()

