import collections


def define(typename, **fields):
    items = list(fields.items())
    # makes it deterministic between python runs.
    items = sorted(items)
    keys = [k for k, _ in items]
    values = [v for _, v in items]
    T = collections.namedtuple(typename, keys)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    prototype = T(*values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


WorldParams = define('WorldParams',
                     randomize_light=False,
                     randomize_material=False,
                     num_substeps=1,
                     # Minimum distance between placed objects.
                     placement_margin=0.0,
                     show_outer_bounds=False,
                     size=(10., 10., 2.5),
                     )
