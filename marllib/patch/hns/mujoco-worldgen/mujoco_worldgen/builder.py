import logging
from collections import OrderedDict
from copy import deepcopy as copy

import numpy as np
from mujoco_py import const, load_model_from_xml, MjSim

from mujoco_worldgen.util.path import worldgen_path
from mujoco_worldgen.objs.obj import Obj
from mujoco_worldgen.parser import unparse_dict, update_mujoco_dict

logger = logging.getLogger(__name__)


class WorldBuilder(Obj):
    classname = 'worldbuilder'  # Used for __repr__

    def __init__(self, world_params, seed):
        self.world_params = copy(world_params)
        self.random_state = np.random.RandomState(seed)
        super(WorldBuilder, self).__init__()
        # Normally size is set during generate() but we are the top level world
        self.size = world_params.size
        # Normally relative_position is set by our parent but we are the root.
        self.relative_position = (0, 0)

    def append(self, obj):
        super(WorldBuilder, self).append(obj, "top")
        return self

    def generate_xml_dict(self):
        ''' Get the mujoco header XML dict. It contains compiler, size and option nodes. '''
        compiler = OrderedDict()
        compiler['@angle'] = 'radian'
        compiler['@coordinate'] = 'local'
        compiler['@meshdir'] = worldgen_path('assets/stls')
        compiler['@texturedir'] = worldgen_path('assets/textures')
        option = OrderedDict()
        option["flag"] = OrderedDict([("@warmstart", "enable")])
        return OrderedDict([('compiler', compiler),
                            ('option', option)])

    def generate_xinit(self):
        return {}  # Builder has no xinit

    def to_xml_dict(self):
        '''
        Generates XML for this object and all of its children.
            see generate_xml() for parameter documentation. Builder
            applies transform to all the children.
        Returns merged xml_dict
        '''
        xml_dict = self.generate_xml_dict()
        assert len(self.markers) == 0, "Can't mark builder object."
        # Then add the xml of all of our children
        for children in self.children.values():
            for child, _ in children:
                child_dict = child.to_xml_dict()
                update_mujoco_dict(xml_dict, child_dict)
        for transform in self.transforms:
            transform(xml_dict)
        return xml_dict

    def get_sim(self):
        self.placements = OrderedDict()
        self.placements["top"] = {"origin": np.zeros(3),
                                  "size": self.world_params.size}
        name_indexes = OrderedDict()
        self.to_names(name_indexes)
        res = self.compile(self.random_state, world_params=self.world_params)
        if not res:
            raise FullVirtualWorldException('Failed to compile world')
        self.set_absolute_position((0, 0, 0))  # Recursively set all positions
        xml_dict = self.to_xml_dict()
        xinit_dict = self.to_xinit()
        udd_callbacks = self.to_udd_callback()
        xml = unparse_dict(xml_dict)

        model = load_model_from_xml(xml)
        sim = MjSim(model, nsubsteps=self.world_params.num_substeps)
        for name, value in xinit_dict.items():
            sim.data.set_joint_qpos(name, value)
        # Places mocap where related bodies are.
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        udd_callbacks = (udd_callbacks or [])
        if udd_callbacks is not None and len(udd_callbacks) > 0:
            def merged_udd_callback(sim):
                ret = {}
                for udd_callback in udd_callbacks:
                    ret.update(udd_callback(sim))
                return ret
            sim.udd_callback = merged_udd_callback
        return sim


class FullVirtualWorldException(Exception):
    def __init__(self, msg=''):
        Exception.__init__(self, "Virtual world is full of objects. " +
                                 "Cannot allocate more of them. " + msg)
