import glob
import os
from mujoco_worldgen.parser.normalize import normalize_value


def parse_arguments(argv):
    '''
    Takes list of arguments and splits them
    to argument that are of form key=value, and dictionary.
    Furhter, cleans arguments (expands *, ~), and
    makes sure that they refer to files, then files
    are local.
    '''
    assert len(argv) >= 1, "At least one argument expected."
    argv = _expand_user_rewrite(argv)
    argv = _expand_wildcard_rewrite(argv)

    argv, kwargs = _extract_kwargs_rewrite(argv)
    _eval_kwargs(kwargs)
    names = argv

    print("\nInferred:")
    print("\tnames: %s" % " ".join(names))
    print("\targuments: %s" % str(kwargs))
    print("\n")

    return names, kwargs


def _expand_wildcard_rewrite(argv):
    '''
    :param argv: list of values
    :return: If arguments contains *, than try to expand it to all fitting files.
    '''
    ret = []
    for arg in argv:
        if "*" in arg:
            new_name = glob.glob(arg)
            assert len(new_name) > 0, "Couldn't find any expansion to the pattern \"%s\"" % arg
            ret += new_name
        else:
            ret.append(arg)
    return ret


def _expand_user_rewrite(argv):
    '''
    :param argv: list of values
    :return: values after the rewrite. If value contains ~ then it's expanded to home directory.
    '''
    ret = []
    for arg in argv:
        if arg[0] == "~":
            arg = os.path.expanduser(arg)
        ret.append(arg)
    return ret


def _extract_kwargs_rewrite(argv):
    '''
    Splits list into dictionary like arguments and remaining arguments.
    :param argv: list of values
    :return: arguments that doesnt look like key=value, and dictionary with remaining arguments.
    '''
    kwargs = {}
    ret = []
    for arg in argv:
        if arg.find("=") > -1:
            pos = arg.find("=")
            key, value = arg[:pos], arg[pos+1:]
            kwargs[key] = normalize_value(value)
        else:
            ret.append(arg)
    return ret, kwargs


def _eval_kwargs(kwargs):
    '''
    Evaluates values which are strings starting with `@`, e.g. "@[]" -> [].
    :param kwargs: dictionary
    :return: the same dictionary but with evaluated values
    '''
    for key, value in kwargs.items():
        if isinstance(value, str) and value[0] == '@':
            kwargs[key] = eval(value[1:])
