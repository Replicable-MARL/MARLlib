import os
import numpy as np
import json
import _jsonnet
from os.path import join
from collections import OrderedDict
from glob import glob

from mujoco_py import load_model_from_xml, load_model_from_mjb, MjSim
from runpy import run_path

from mujoco_worldgen import Env
from mujoco_worldgen.util.path import worldgen_path
from mujoco_worldgen.util.types import extract_matching_arguments
from mujoco_worldgen.parser import parse_file, unparse_dict


def get_function(fn_data):
    name = fn_data['function']
    extra_args = fn_data['args']
    module_path, function_name = name.rsplit(':', 1)
    result = getattr(__import__(module_path, fromlist=(function_name,)), function_name)
    if len(extra_args) > 0:
        def result_wrapper(*args, **kwargs):
            actual_kwargs = extra_args.copy()
            actual_kwargs.update(kwargs)
            return result(*args, **actual_kwargs)
        return result_wrapper
    else:
        return result


def load_env(pattern, core_dir=worldgen_path(), envs_dir='examples', xmls_dir='xmls',
             return_args_remaining=False, **kwargs):
    """
    Flexible load of an environment based on `pattern`.
    Passes args to make_env().
    :param pattern: tries to match environment to the pattern.
    :param core_dir: Absolute path to the core code directory for the project containing
        the environments we want to examine. This is usually the top-level git repository
        folder - in the case of the mujoco-worldgen repo, it would be the 'mujoco-worldgen'
        folder.
    :param envs_dir: relative path (from core_dir) to folder containing all environment files.
    :param xmls_dir: relative path (from core_dir) to folder containing all xml files.
    :param return_remaining_kwargs: returns arguments from kwargs that are not used.
    :param kwargs: arguments passed to the environment function.
    :return: mujoco_worldgen.Env
    """
    # Loads environment based on XML.
    env = None
    args_remaining = {}
    if pattern.endswith(".xml"):
        if len(kwargs) > 0:
            print("Not passing any argument to environment, "
                  "because environment is loaded from XML. XML doesn't "
                  "accept any extra input arguments")

        def get_sim(seed):
            model = load_model_from_path_fix_paths(xml_path=pattern)
            return MjSim(model)
        env = Env(get_sim=get_sim)
    # Loads environment based on mjb.
    elif pattern.endswith(".mjb"):
        if len(kwargs) != 0:
            print("Not passing any argument to environment, "
                  "because environment is loaded from MJB. MJB doesn't "
                  "accept any extra input arguments")

        def get_sim(seed):
            model = load_model_from_mjb(pattern)
            return MjSim(model)
        env = Env(get_sim=get_sim)
    # Loads environment from a python file
    elif pattern.endswith("py") and os.path.exists(pattern):
        
        print("Loading env from the module: %s" % pattern)
        module = run_path(pattern)
        make_env = module["make_env"]
        args_to_pass, args_remaining = extract_matching_arguments(make_env, kwargs)
        env = make_env(**args_to_pass)
        
    elif pattern.endswith(".jsonnet") and os.path.exists(pattern):
        env_data = json.loads(_jsonnet.evaluate_file(pattern))
        make_env = get_function(env_data['make_env'])
        args_to_pass, args_remaining = extract_matching_arguments(make_env, kwargs)
        env = make_env(**args_to_pass)
    else:
        # If couldn't load based on easy search, then look
        # into predefined subdirectories.
        matching = (glob(join(core_dir, envs_dir, "**", "*.py"), recursive=True) +
                    glob(join(core_dir, xmls_dir, "**", "*.xml"), recursive=True))
        matching = [match for match in matching if match.find(pattern) > -1]
        matching = [match for match in matching if not os.path.basename(match).startswith('test_')]
        assert len(matching) < 2, "Found multiple environments matching %s" % str(matching)
        if len(matching) == 1:
            return load_env(matching[0], return_args_remaining=return_args_remaining, **kwargs)
    if return_args_remaining:
        return env, args_remaining
    else:
        return env


def load_model_from_path_fix_paths(xml_path, zero_gravity=True):
    """
    Loads model from XML path. Ensures that
    all assets are locally available. If needed might rename
    paths.

    :param xml_path: path to xml file
    :param zero_gravity: if true, zero gravity in model
    """
    xml_dict = parse_file(xml_path, enforce_validation=False)

    if zero_gravity:
        # zero gravity so that the object doesn't fall down
        option = xml_dict.setdefault('option', OrderedDict())
        option['@gravity'] = np.zeros(3)
    xml = unparse_dict(xml_dict)
    model = load_model_from_xml(xml)
    return model
