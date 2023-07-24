from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.objs.obj import Obj
import numpy as np
from collections import OrderedDict


class Floor(Obj):
    '''
    Floor() is essentially a special box geom used as the base of experiments.
    It has no joints, so is essentially an immovable object.
    Placement is calculated in a fixed position, and encoded in XML,
        as opposed to in qpos, which other objects use.
    '''
    @store_args
    def __init__(self, geom_type='plane'):
        super(Floor, self).__init__()

    def generate(self, random_state, world_params, placement_size):
        top = OrderedDict(origin=(0, 0, 0), size=placement_size)
        self.placements = OrderedDict(top=top)
        self.size = np.array([placement_size[0], placement_size[1], 0.0])

    def generate_xml_dict(self):
        # Last argument in size is visual mesh resolution (it's not height).
        # keep it high if you want rendering to be fast.
        pos = self.absolute_position
        pos[0] += self.size[0] / 2.0
        pos[1] += self.size[1] / 2.0
        geom = OrderedDict()

        geom['@name'] = self.name
        geom['@pos'] = pos
        if self.geom_type == 'box':
            geom['@size'] = np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.000001])
            geom['@type'] = 'box'
        elif self.geom_type == 'plane':
            geom['@size'] = np.array([self.size[0] / 2.0, self.size[1] / 2.0, 1.0])
            geom['@type'] = 'plane'
        else:
            raise ValueError("Invalid geom_type: " + self.geom_type)
        geom['@condim'] = 3
        geom['@name'] = self.name

        # body is necessary to place sites.
        body = OrderedDict()
        body["@name"] = self.name
        body["@pos"] = pos

        worldbody = OrderedDict([("geom", [geom]),
                                 ("body", [body])])

        xml_dict = OrderedDict(worldbody=worldbody)
        return xml_dict
