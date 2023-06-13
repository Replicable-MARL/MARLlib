from collections import OrderedDict

from mujoco_worldgen.objs.obj import Obj
from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.obj_util import get_body_xml_node
from mujoco_worldgen.util.path import worldgen_path
import stl
import os
import numpy as np


class ObjFromSTL(Obj):
    """
        Creates an object based on an STL file.
        Read documentation for the worldgen Obj class for more info.
    """
    @store_args
    def __init__(self, path, name=None):
        super(ObjFromSTL, self).__init__()

    def generate(self, random_state, world_params, placement_size):
        if os.path.exists(self.path):
            self.local_path = self.path
        else:
            self.local_path = worldgen_path("assets/stls", self.path)
        if not isinstance(self.local_path, list):
            self.local_path = [self.local_path]
        self.objs = []
        max_ = np.zeros(3) - np.inf
        min_ = np.zeros(3) + np.inf
        for path in self.local_path:
            obj = stl.mesh.Mesh.from_file(path)
            for i in range(3):
                max_[i] = max(max_[i], obj.max_[i])
                min_[i] = min(min_[i], obj.min_[i])
            self.objs.append(obj)
        self.placements = OrderedDict()
        self.size = max_ - min_
        self.min_ = min_

    def generate_xml_dict(self):
        body = get_body_xml_node(self.name, use_joints=True)
        for jnt in body["joint"]:
            jnt["@damping"] = 0.1
        xml_dict = OrderedDict()
        xml_dict['worldbody'] = OrderedDict(body=[body])
        xml_dict["asset"] = OrderedDict(mesh=[])

        self.body = body  # Save for use in generate_xinit()
        body['geom'] = []
        for idx, path in enumerate(self.local_path):
            geom = OrderedDict()
            name = self.name + "_" + str(idx)
            mesh = OrderedDict([("@name", name),
                                ("@file", path)])
            xml_dict["asset"]["mesh"].append(mesh)
            geom['@type'] = "mesh"
            geom['@condim'] = 6
            geom['@name'] = name
            geom['@mesh'] = name
            geom['@pos'] = -self.min_ - self.size / 2
            body['geom'].append(geom)
        body["@pos"] = self.size / 2
        return xml_dict
