from collections import OrderedDict
from decimal import getcontext
from os.path import abspath, dirname, join, exists
from mujoco_worldgen.transforms import closure_transform

import numpy as np
import xmltodict
import os

from mujoco_worldgen.util.types import accepts, returns
from mujoco_worldgen.util.path import worldgen_path
from mujoco_worldgen.parser.normalize import normalize, stringify

getcontext().prec = 4

'''
This directory should contain all XML string processing.
No other files should be manually converting types for XML processing.
API:
    parse_file() - takes in path to mujoco file and returns normalized dictionary
    unparse_dict() - takes in an xml dictionary and returns an XML string
NOTE FOR TRANSFORMS:
    The internal xml_dict layout passed into transforms is the one returned by
    normalize() -- see that docstring for more details on its layout.
Every other method should be considered internal!
'''


@accepts(str, bool)
@returns(OrderedDict)
def parse_file(xml_path, enforce_validation=True):
    '''
    Reads xml from xml_path, consolidates all includes in xml, and returns
    a normalized xml dictionary.  See preprocess()
    '''
    # TODO: use XSS or DTD checking to verify XML structure
    with open(xml_path) as f:
        xml_string = f.read()

    xml_doc_dict = xmltodict.parse(xml_string.strip())
    assert 'mujoco' in xml_doc_dict, "XML must contain <mujoco> node"
    xml_dict = xml_doc_dict['mujoco']
    assert isinstance(xml_dict, OrderedDict), \
        "Invalid node type {}".format(type(xml_dict))
    preprocess(xml_dict, xml_path, enforce_validation=enforce_validation)
    return xml_dict


@accepts(OrderedDict)
@returns(str)
def unparse_dict(xml_dict):
    '''
    Convert a normalized XML dictionary into a XML string.  See stringify().
    Note: this modifies xml_dict in place to have strings instead of values.
    '''
    stringify(xml_dict)
    xml_doc_dict = OrderedDict(mujoco=xml_dict)
    return xmltodict.unparse(xml_doc_dict, pretty=True)


@accepts(OrderedDict, str, bool)
def preprocess(xml_dict, root_xml_path, enforce_validation=True):
    '''
    All the steps to turn XML into Worldgen readable form:
    - normalize: changes strings to floats / vectors / bools, and
                 turns consistently nodes to OrderedDict and List
    - name_meshes: some meshes are missing names. Here we give default names.
    - rename_defaults: some defaults are global, we give them names so
                       they won't be anymore.
    - extract_includes: recursively, we extract includes and merge them.
    - validate: we apply few final checks on the structure.
    '''
    normalize(xml_dict)
    set_absolute_paths(xml_dict, root_xml_path)
    extract_includes(xml_dict, root_xml_path, enforce_validation=enforce_validation)
    if enforce_validation:
        validate(xml_dict)


@accepts(OrderedDict, str)
def set_absolute_paths(xml_dict, root_xml_path):
    dirnames = ["@meshdir", "@texturedir"]
    if "compiler" in xml_dict:
        for drname in dirnames:
            if drname in xml_dict["compiler"]:
                asset_dir = worldgen_path('assets') + '/'
                path = xml_dict["compiler"][drname]
                if path[0] != "/":
                    relative_path = os.path.dirname(root_xml_path) + "/" + path
                    xml_dict["compiler"][drname] = os.path.abspath(relative_path)
                elif path.find(asset_dir) > -1:
                    xml_dict["compiler"][drname] = worldgen_path(
                        'assets', path.split(asset_dir)[-1])


@accepts(OrderedDict, str, bool)
def extract_includes(xml_dict, root_xml_path, enforce_validation=True):
    '''
    extracts "include" xmls and substitutes them.
    '''
    def transform_include(node):
        if "include" in node:
            if isinstance(node["include"], OrderedDict):
                node["include"] = [node["include"]]
            include_xmls = []
            for include_dict in node["include"]:
                include_path = include_dict["@file"]
                if not exists(include_path):
                    include_path = join(dirname(abspath(root_xml_path)), include_path)
                assert exists(include_path), "Cannot include file: %s" % include_path
                with open(include_path) as f:
                    include_string = f.read()
                include_xml = xmltodict.parse(include_string.strip())
                closure_transform(transform_include)(include_xml)
                assert "mujocoinclude" in include_xml, "Missing <mujocoinclude>."
                include_xmls.append(include_xml["mujocoinclude"])
            del node["include"]
            for include_xml in include_xmls:
                preprocess(include_xml, root_xml_path, enforce_validation=enforce_validation)
                update_mujoco_dict(node, include_xml)
    closure_transform(transform_include)(xml_dict)


@accepts(OrderedDict, OrderedDict)
@returns(None.__class__)
def update_mujoco_dict(dict_a, dict_b):
    '''
    Update mujoco dict_a with the contents of another mujoco dict_b.
    '''
    other = (str, int, float, np.ndarray, tuple)
    for key, value in dict_b.items():
        if key not in dict_a:
            dict_a[key] = value
        elif isinstance(dict_a[key], list):
            assert isinstance(value, list), "Expected %s to be a list" % value
            dict_a[key] += value
        elif isinstance(value, other):
            assert(isinstance(dict_a[key], other))
            assert dict_a[key] == value, "key=%s\n,Trying to merge dictionaries. " \
                                         "They don't agree on value: %s vs %s" % (key, dict_a[key], value)
        else:
            assert isinstance(dict_a[key], OrderedDict), "dict_a = %s\nkey=%s\nExpected dict_a[key] to be a OrderedDict." % (dict_a, key)
            assert(isinstance(value, OrderedDict))
            update_mujoco_dict(dict_a[key], value)


@accepts(OrderedDict)
def validate(xml_dict):
    '''
    If we make assumptions elsewhere in XML processing, then they should be
        enforced here.
    '''
    # Assumption: radians for angles, "xyz" euler angle sequence, etc.

    values = {'@coordinate': 'local',
              '@angle': 'radian',
              '@eulerseq': 'xyz'}
    for key, value in values.items():
        if key in xml_dict:
            assert value == xml_dict[key], 'Invalid value for \"%s\". We support only \"%s\"' % (key, value)

    # Assumption: all meshes have name
    if "asset" in xml_dict and "mesh" in xml_dict["asset"]:
        for mesh in xml_dict["asset"]["mesh"]:
            assert "@name" in mesh, "%s is missing name" % mesh

    # Assumption: none all the default classes is global.
    if "default" in xml_dict:
        for key, value in xml_dict["default"].items():
            assert key == "default", "Dont use global variables in default %s %s" % (key, value)

    # Assumption: all joints have name.
    def assert_joint_names(node):
        if "joint" in node:
            for joint in node["joint"]:
                assert "@name" in joint, "Missing name for %s" % joint

    if "worldbody" in xml_dict:
        closure_transform(assert_joint_names)(xml_dict["worldbody"])
