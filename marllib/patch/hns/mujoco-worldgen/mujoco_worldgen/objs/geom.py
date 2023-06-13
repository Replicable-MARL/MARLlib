import numpy as np
from collections import OrderedDict

from mujoco_worldgen.objs.obj import Obj
from mujoco_worldgen.util.obj_util import establish_size, get_body_xml_node
from mujoco_worldgen.util.types import store_args


class Geom(Obj):

    @store_args
    def __init__(self, geom_type,
                 min_size=None,
                 max_size=None,
                 name=None,
                 rgba=None):
        super(Geom, self).__init__()

    def generate(self, random_state, world_params, placement_size):
        min_size, max_size = establish_size(self.min_size, self.max_size)
        # TODO: Current worldgen doesn't respect height.
        for i in range(2):
            max_size[i] = min(max_size[i], placement_size[i])
        self.placements = OrderedDict()
        if self.geom_type == "box":
            self.size = min_size + (max_size - min_size) * \
                random_state.uniform(size=3)
            top_height = placement_size[2] - self.size[2]
            top = OrderedDict(origin=(0, 0, self.size[2]),
                              size=(self.size[0], self.size[1], top_height))
            self.placements['top'] = top
        elif self.geom_type == "sphere":
            min_size = np.max(min_size)
            max_size = np.min(max_size)
            assert(min_size <= max_size)
            self.size = np.ones(
                3) * (min_size + (max_size - min_size) * random_state.uniform(size=1))
        elif self.geom_type == "cylinder":
            self.size = min_size[:2] + (max_size[:2] - min_size[:2]) * \
                random_state.uniform(size=2)
            self.size = np.array([self.size[0], self.size[0], self.size[1]])

    def generate_xml_dict(self):
        '''
        Generate XML dict needed for MuJoCo model.
        Returns a dictionary with keys as names of top-level nodes:
            e.g. 'worldbody', 'materials', 'assets'
        '''
        body = get_body_xml_node(self.name, use_joints=True)
        geom = OrderedDict()
        geom['@size'] = self.size * 0.5
        body['@pos'] = self.size * 0.5
        if self.geom_type == 'cylinder':
            # Mujoco expects only radius and half-length
            geom['@size'] = [geom['@size'][0], geom['@size'][2]]
        geom['@type'] = self.geom_type
        geom['@condim'] = 3
        geom['@name'] = self.name
        if self.rgba is not None:
            geom['@rgba'] = self.rgba
        body['geom'] = [geom]
        xml_dict = OrderedDict()
        xml_dict['worldbody'] = OrderedDict(body=[body])
        return xml_dict
