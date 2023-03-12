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

import yaml
import os
import collections

algo_type_dict = {
    "IL": ["a2c", "a3c", "pg", "ddpg", "trpo", "ppo"],
    "VD": ["vda2c", "vdppo", "facmac", "iql", "vdn", "qmix"],
    "CC": ["maa2c", "maddpg", "mappo", "matrpo", "happo", "hatrpo", "coma"]
}


def merge_default_and_customized_and_check(default_dict, customized_dict):
    if customized_dict and isinstance(customized_dict, dict):
        for k in customized_dict.keys():
            if k not in default_dict:
                raise ValueError("{} illegal, not in default config".format(k))
            else:  # update
                default_dict[k] = customized_dict[k]

    return default_dict


def merge_default_and_customized(default_dict, customized_dict):
    if customized_dict and isinstance(customized_dict, dict):
        for key, value in customized_dict.items():
            if key in default_dict:
                default_dict[key] = value

    return default_dict


def check_algo_type(algo_name):
    for key in algo_type_dict.keys():
        if algo_name in algo_type_dict[key]:
            return key
    raise ValueError("{} current not supported".format(algo_name))


def get_model_config(arg_name):
    with open(os.path.join(os.path.dirname(__file__), "models/configs", "{}.yaml".format(arg_name)),
              "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(arg_name, exc)
    return config_dict


def get_config(params, arg_name, info=None):
    config_name = None

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if "algo" in arg_name:
        if "--finetuned" in params:
            path = "algos/hyperparams/finetuned/{}".format(info["env"])
        else:
            path = "algos/hyperparams/common"

    elif "env" in arg_name:
        path = "../envs/base_env/config"

    else:
        raise ValueError()

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), path, "{}.yaml".format(config_name)),
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
