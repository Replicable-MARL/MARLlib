from os.path import abspath, dirname, join

WORLDGEN_ROOT_PATH = abspath(join(dirname(__file__), '..', '..'))


def worldgen_path(*args):
    """
    Returns an absolute path from a path relative to the mujoco_worldgen repository
    root directory.
    """
    return join(WORLDGEN_ROOT_PATH, *args)
