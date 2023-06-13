from collections import OrderedDict
from mujoco_worldgen.util.types import accepts, returns
from mujoco_worldgen.parser.const import list_types, float_arg_types
import numpy as np
from decimal import Decimal, getcontext
import ast
import re

getcontext().prec = 10

'''
This methods are used internally by parser.py
Internal notes:
    normalize() - in-place normalizes and converts an xml dictionary
        see docstring for notes about what types are converted
    stringify() - in-place de-normalizes, and converts all values to strings
        see docstring for notes
    normalize_*() - return normal forms of values (numbers, vectors, etc)
        raise exception if input value cannot be converted
'''


@accepts(OrderedDict)
def normalize(xml_dict):
    '''
    The starting point is a dictionary of the form returned by xmltodict.
    See that module's documentation here:
        https://github.com/martinblech/xmltodict
    Normalize a mujoco model XML:
        - some nodes have OrderDict value (mostly top-lever such as worldbody)
          some nodes have list values (mostly lower level). Check const.py
          for more information.
        - parameters ('@name', etc) never have list or OrderedDict values
        - "true" and "false" are converted to bool()
        - numbers are converted to floats
        - vectors are converted to np.ndarray()
    Note: stringify() is the opposite of this, and converts everything back
        into strings in preparation for unparse_dict().
    '''
    # As a legacy, previously many of our XMLs had an unused model name.
    # This removes it (as part of annotate) and can be phased out eventually.
    if '@model' in xml_dict:
        del xml_dict['@model']
    for key, value in xml_dict.items():
        if isinstance(value, OrderedDict):
            # There is one exception.
            # <default> symbol occurs twice.
            # Once as OrderDict (top-level), once as list (lower-level).
            if key == "default":
                if "@class" in value:
                    xml_dict[key] = [value]
            elif key in list_types:
                xml_dict[key] = [value]
            normalize(value)
            continue
        if isinstance(value, list):
            for child in value:
                normalize(child)
            continue
        if isinstance(value, str):
            xml_dict[key] = normalize_value(value)
            # sometimes data is stored as int when it's float.
            # We make a conversion here.
            if key in float_arg_types:
                if isinstance(xml_dict[key], int):
                    xml_dict[key] = float(xml_dict[key])
                elif isinstance(xml_dict[key], np.ndarray):
                    xml_dict[key] = xml_dict[key].astype(np.float64)


@accepts((int, float, np.float32, np.float64, np.int64))
@returns(str)
def num2str(num):
    ret = "%g" % Decimal("%.6f" % num)
    if ret == "-0":
        return "0"
    else:
        return ret


@accepts((np.ndarray, tuple, list))
@returns(str)
def vec2str(vec):
    return " ".join([num2str(v) for v in vec])


@returns(bool)
def is_normalizeable(normalize_function, value):
    '''
    Wraps a normalize_*() function, and returns True if value can be
        normalized by normalize_function, otherwise returns False.
    '''
    try:
        normalize_function(value)
        return True
    except:
        return False


def normalize_numeric(value):
    ''' Normalize a numeric value into a float. '''
    if isinstance(value, (float, int, np.float64, np.int64)):
        return value
    if isinstance(value, (str, bytes)):
        f = float(value)
        if f == int(f):  # preferentially return integers if equal
            return int(f)
        return f
    raise ValueError('Cannot convert {} to numeric'.format(value))


@accepts((np.ndarray, list, tuple, str))
def normalize_vector(value):
    ''' Normalize a vector value to a np.ndarray(). '''
    if isinstance(value, np.ndarray):
        return value
    if (isinstance(value, (list, tuple)) and len(value) > 0 and
            is_normalizeable(normalize_numeric, value[0])):
        return np.array(value)
    if isinstance(value, str):
        # Split on spaces, filter empty, convert to numpy array
        if "," in value or re.search("\[.*\]", value) is not None:
            return np.array(ast.literal_eval(value))
        else:
            split = value.split()
            return np.array([normalize_numeric(v) for v in split])
    raise ValueError('Cannot convert {} to vector'.format(value))


def normalize_boolean(value):
    ''' Normalize a boolean value to a bool(). '''
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower().strip() == 'true':
            return True
        if value.lower().strip() == 'false':
            return False
    raise ValueError('Cannot convert {} to boolean'.format(value))


def normalize_none(value):
    ''' Normalize a none string value to a None. '''
    if isinstance(value, None.__class__):
        return value
    if isinstance(value, str):
        if value.lower().strip() == 'none':
            return None
    raise ValueError('Cannot convert {} to None'.format(value))


def normalize_string(value):
    ''' Normalize a string value. '''
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return value.strip()
    raise ValueError('Cannot convert {} to string'.format(value))


def normalize_value(value):
    ''' Return the normalized version of a value by trying normalize_*(). '''
    if value is None:
        return None
    for normalizer in (normalize_numeric,
                       normalize_vector,
                       normalize_none,
                       normalize_boolean,
                       normalize_string):
        try:
            return normalizer(value)
        except:
            continue
    raise ValueError('Cannot normalize {}: {}'.format(type(value), value))


@accepts((OrderedDict, list))
def stringify(xml_dict):
    '''
    De-normalize xml dictionary (or list), converting all pythonic values (arrays, bools)
        into strings that will be used in the final XML.
    This is the opposite of normalize().
    '''
    if isinstance(xml_dict, OrderedDict):
        enumeration = list(xml_dict.items())
    elif isinstance(xml_dict, list):
        enumeration = enumerate(xml_dict)

    for key, value in enumeration:
        # Handle a list of nodes to stringify
        if isinstance(value, list):
            if len(value) == 0:
                del xml_dict[key]
            else:
                if sum([isinstance(v, (int, float, np.float32, np.int)) for v in value]) == len(value):
                    xml_dict[key] = vec2str(value)
                else:
                    stringify(value)
        elif isinstance(value, OrderedDict):
            stringify(value)
        elif isinstance(value, (np.ndarray, tuple)):
            xml_dict[key] = vec2str(value)
        elif isinstance(value, float):
            xml_dict[key] = num2str(value)  # format with fixed decimal places
        elif isinstance(value, bool):  # MUST COME BEFORE int() CHECK
            xml_dict[key] = str(value).lower()  # True -> 'true', etc.
        elif isinstance(value, int):  # isinstance(True, int) -> True.  SAD!
            xml_dict[key] = str(value)  # Format without decimal places
        elif isinstance(value, str):
            pass  # Value is already fine
        elif value is None:
            pass
        else:
            raise ValueError(
                'Bad type for key {}: {}'.format(key, type(value)))
