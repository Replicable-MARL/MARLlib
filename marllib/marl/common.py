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
from typing import Dict

algo_type_dict = {
    "IL": ["ia2c", "iddpg", "itrpo", "ippo"],
    "VD": ["vda2c", "vdppo", "facmac", "iql", "vdn", "qmix"],
    "CC": ["maa2c", "maddpg", "mappo", "matrpo", "happo", "hatrpo", "coma"]
}


def dict_update(target_dict: Dict, new_dict: Dict, check: bool = False) -> Dict:
    """
    update target dict with new dict
    Args:
        :param target_dict: name of the environment
        :param new_dict: name of the algorithm
        :param check: whether a new key is allowed to add into target_dict

    Returns:
        Dict: updated dict
    """
    if new_dict and isinstance(new_dict, dict):
        for key, value in new_dict.items():
            if check:
                if key not in target_dict:
                    raise ValueError("{} illegal, not in default config".format(key))
                else:  # update
                    target_dict[key] = value
            else:
                target_dict[key] = value

    return target_dict


def recursive_dict_update(target_dict: Dict, new_dict: Dict) -> Dict:
    """
    recursively update target dict with new dict
    Args:
        :param target_dict: name of the environment
        :param new_dict: name of the algorithm

    Returns:
        Dict: updated dict
    """
    for k, v in new_dict.items():
        if isinstance(v, collections.Mapping):
            target_dict[k] = recursive_dict_update(target_dict.get(k, {}), v)
        else:
            target_dict[k] = v
    return target_dict


def check_algo_type(algo_name: str) -> str:
    """
    check algorithm learning style from 1. il, 2. cc, 3. vd
    Args:
        :param algo_name: name of the algorithm

    Returns:
        str: learning style from 1. il, 2. cc, 3. vd
    """
    for key in algo_type_dict.keys():
        if algo_name in algo_type_dict[key]:
            return key
    raise ValueError("{} current not supported".format(algo_name))


def get_model_config(model_arch: str) -> Dict:
    """
    read model config
    Args:
        :param model_arch: type of the model

    Returns:
        Dict: model config dict
    """
    with open(os.path.join(os.path.dirname(__file__), "models/configs", "{}.yaml".format(model_arch)),
              "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(model_arch, exc)
    return config_dict
