""" Creates an object based on MujocoXML. XML has to have annotation such as
      - annotation:outer_bound : defines box that spans the entire object.
    Moreover, its left lower corner should be located at (0, 0, 0)
"""

import glob
import os
from collections import OrderedDict

import numpy as np

from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.path import worldgen_path
from mujoco_worldgen.parser import parse_file
from mujoco_worldgen.util.obj_util import get_name_index, recursive_rename
from mujoco_worldgen.objs.obj import Obj


class ObjFromXML(Obj):
    """
    Creates an object based on MujocoXML. XML has to have annotation such as
        - annotation:outer_bound : defines box that spans the entire object.
    Moreover, its left lower corner should be located at (0, 0, 0)
    """
    @store_args
    def __init__(self, model_path, name=None, default_qpos=None):
        super(ObjFromXML, self).__init__()

    def generate(self, random_state, world_params, placement_size):
        # Only do this once, because it sometimes picks object at random
        self.xml_path = self._generate_xml_path(random_state)
        self.xml = parse_file(self.xml_path)
        self.placements = OrderedDict()
        bodies = []
        for body in self.xml["worldbody"]["body"]:
            name = body.get('@name', '')
            if name.startswith('annotation:'):
                # Placement annotation, for example the insides of shelves,
                # or outer_bound, which determines size and "top" placement.
                assert '@pos' in body, "Annotation %s must have pos" % name
                assert 'geom' in body, "Annotation %s must have geom" % name
                assert len(body['geom']) == 1, "%s must have 1 geom" % name
                geom = body['geom'][0]
                assert geom.get('@type') == 'box', "%s must have box" % name
                assert '@size' in geom, "%s geom must have size" % name
                if '@pos' in geom:
                    # Worldgen places objects by moving qpos (slide joints)
                    # to put them in position, and their final position is:
                    #   qpos + pos + parent_pos + ...
                    # In order for objects to end up where worldgen wants them,
                    # all of the pos + parent_pos + ... have to equal zero.
                    # Otherwise the offsets get messed up.
                    assert np.array_equal(geom['@pos'], np.zeros(3)), \
                        "%s: Set pos on body instead of geom" % name
                size = geom['@size'] * 2  # given as halfsize
                origin = body['@pos'] - (size / 2)  # given as center coord
                placement_name = name[len('annotation:'):]
                if placement_name == 'outer_bound':
                    # Note: "top" placement is not automatically created
                    # Must be explicitly added in XML
                    # bin/annotate.py --suggestions will show possible ones
                    self.size = size
                    if world_params.show_outer_bounds:
                        bodies.append(body)
                    continue
                placement = OrderedDict(size=size, origin=origin)
                self.placements[placement_name] = placement

        for body in self.xml["worldbody"]["body"]:
            name = body.get('@name', '')
            if not name.startswith("annotation:"):  # Not an annotation, must be a main body
                if self.name is not None:
                    body_name = self.name
                    if name:
                        body_name += ":" + name
                    body['@name'] = body_name
                    body["@pos"] = body["@pos"]
                bodies.append(body)
        self.xml['worldbody']['body'] = bodies

    def add_joints(self, body):
        joint_names = []
        if 'joint' not in body:
            body['joint'] = []
        if isinstance(body['joint'], OrderedDict):
            body['joint'] = [body['joint']]
        for i, slide_axis in enumerate(np.eye(3)):
            found = False
            for joint in body['joint']:
                if not isinstance(joint, OrderedDict):
                    continue
                if joint.get('@type') != 'slide':
                    continue
                if '@axis' not in joint:
                    continue
                axis = joint['@axis']
                if np.linalg.norm(slide_axis - axis) < 1e-6:
                    joint_names.append(joint['@name'])
                    found = True
                    break  # Found axis
            if not found:  # add this joint
                slide = OrderedDict()
                joint_name = self.name + ':slide%d' % i
                slide['@name'] = joint_name
                slide['@type'] = 'slide'
                slide['@axis'] = slide_axis
                slide['@damping'] = '0.01'
                slide['@pos'] = np.zeros(3)
                body['joint'].append(slide)
                joint_names.append(joint_name)
        return joint_names

    def generate_name(self, name_indexes):
        if self.name is None:
            if self.model_path.split("/")[0] == "robot":
                assert self.name is None or self.name == "robot", \
                    "Detected robot XML. " \
                    "Robot should be named \"robot\". Abording."
                name = "robot"
            else:
                name = self.model_path.replace('/', '_')
            self.name = get_name_index(name_indexes, name)

    def generate_xml_dict(self):
        '''
        Generate XML DOM nodes needed for MuJoCo model.
            doc - XML Document, used to create elements/nodes
            name_indexes - dictionary to keep track of names,
                see get_name_index() for internals
        Returns a dictionary with keys as names of top-level nodes:
            e.g. 'worldbody', 'materials', 'assets'
        And the values are lists of XML DOM nodes
        '''
        # Iterate over all names inside and prepend self.name
        recursive_rename(self.xml, self.name)
        main_body = None
        worldbody = self.xml["worldbody"]
        bodies = worldbody["body"]
        for body in bodies:
            name = body.get('@name', '')  # Might not be present in main body
            if "annotation" not in name and not body.get('@mocap'):
                assert main_body is None, "We support only a single main body."
                main_body = body
        for rot in ('@euler', '@quat'):
            assert rot not in main_body, 'We dont support rotations in the main body.'\
                                         'Please move it inward.'
        self.add_joints(main_body)
        return self.xml

    def _get_xml_dir_path(self, *args):
        '''
        If you want to use custom XMLs, subclass this class and overwrite this
        method to return the path to your 'xmls' folder
        '''
        return worldgen_path('assets/xmls', *args)

    def _generate_xml_path(self, random_state=None):
        '''Separated because some subclasses need to override just this'''
        if random_state is None:
            random_state = np.random.RandomState(0)
        xml_path = self._get_xml_dir_path(self.model_path)
        if not xml_path.endswith(".xml"):
            # Didn't find it, go to a subdirectory
            if not os.path.isfile(os.path.join(xml_path, "main.xml")):
                dirs = glob.glob(os.path.join(xml_path, '*'))
                assert dirs, "Failed to find dirs matching {}".format(xml_path)
                xml_path = dirs[random_state.randint(0, len(dirs))]
            xml_path = os.path.join(xml_path, "main.xml")
        return xml_path
