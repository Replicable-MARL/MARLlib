import numpy as np
import hashlib
from collections import OrderedDict
from mujoco_worldgen.objs.obj import Obj
from mujoco_worldgen.util.types import store_args


class Material(Obj):
    placeable = False

    @store_args
    def __init__(self,
                 random=True,
                 rgba=None,
                 texture=None,
                 texture_type=None,
                 grid_layout=None,
                 grid_size=None):
        super(Material, self).__init__()

    def generate(self, random_state, world_params, placement_size=None):
        if not world_params.randomize_material:
            deterministic_seed = int(hashlib.sha1(
                self.name.encode()).hexdigest(), 16)
            random_state = np.random.RandomState(deterministic_seed % 100000)
        choice = random_state.randint(0, 3)
        self.xml_dict = None
        if self.texture is not None:
            self.xml_dict = self._material_texture(
                random_state, self.texture, self.texture_type,
                self.grid_layout, self.grid_size, self.rgba)
        elif self.rgba is not None:
            self.xml_dict = self._material_rgba(random_state, self.rgba)
        elif self.xml_dict is None:
            self.xml_dict = [self._material_rgba,
                             self._material_checker,
                             self._material_random][choice](random_state)
        self.xml_dict = OrderedDict(asset=self.xml_dict)

    def generate_xml_dict(self):
        return self.xml_dict

    def _material_rgba(self, random_state, rgba=None):
        material_attrs = OrderedDict([('@name', self.name),
                                      ('@specular', 0.1 + 0.2 *
                                       random_state.uniform()),
                                      ('@shininess', 0.1 + 0.2 *
                                       random_state.uniform()),
                                      ('@reflectance', 0.1 + 0.2 * random_state.uniform())])
        if rgba is None:
            material_attrs['@rgba'] = 0.1 + 0.8 * random_state.uniform(size=4)
            material_attrs['@rgba'][3] = 1.0
        elif isinstance(rgba, tuple) and len(rgba) == 2:
            material_attrs['@rgba'] = random_state.uniform(rgba[0], rgba[1])
        else:
            material_attrs['@rgba'] = rgba
        return OrderedDict(material=[material_attrs])

    def _material_checker(self, random_state):
        texture_attr = OrderedDict([('@name', "texture_" + self.name),
                                    ('@builtin', 'checker'),
                                    ('@height', random_state.randint(5, 100)),
                                    ('@width', random_state.randint(5, 100)),
                                    ('@type', '2d'),
                                    ('@rgb1', [0, 0, 0])])
        texture_attr['@rgb2'] = 0.1 + 0.8 * random_state.uniform(size=3)
        xml_dict = OrderedDict(texture=[texture_attr])
        texrepeat = [random_state.randint(
            5, 100), random_state.randint(5, 100)]
        xml_dict["material"] = [OrderedDict([('@name', self.name),
                                             ('@texrepeat', texrepeat),
                                             ('@texture', "texture_" + self.name)])]
        return xml_dict

    def _material_random(self, random_state):
        random = 0.1 + 0.8 * random_state.uniform()
        texture_attr = OrderedDict([('@name', "texture_" + self.name),
                                    ('@builtin', 'flat'),
                                    ('@mark', 'random'),
                                    ('@type', '2d'),
                                    ('@height', 2048),
                                    ('@width', 2048),
                                    ('@rgb1', [1, 1, 1]),
                                    ('@rgb2', [1, 1, 1]),
                                    ('@random', random)])
        material = OrderedDict([('@name', self.name),
                                ('@texture', "texture_" + self.name)])
        xml_dict = OrderedDict([('texture', [texture_attr]),
                                ('material', [material])])
        return xml_dict

    def _material_texture(self, random_state, texture, texture_type=None,
                          grid_layout=None, grid_size=None, rgba=None):
        texture_attr = OrderedDict([
            ('@name', "texture_" + self.name),
            ('@type', '2d'),
            ('@builtin', 'none'),
            ('@file', texture),
        ])

        if texture_type is None:
            texture_type = "cube"
        texture_attr["@type"] = texture_type
        if texture_type == "cube":
            texture_attr["@gridlayout"] = '.U..LFRB.D..' if grid_layout is None else grid_layout
            texture_attr["@gridsize"] = '3 4' if grid_size is None else grid_size

        material = OrderedDict([
            ('@name', self.name),
            ('@texture', "texture_" + self.name),
        ])

        if rgba is not None:
            material['@rgba'] = rgba

        return OrderedDict([
            ('texture', [texture_attr]),
            ('material', [material]),
        ])
