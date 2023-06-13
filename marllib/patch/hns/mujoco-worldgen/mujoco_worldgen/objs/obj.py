import inspect
from collections import OrderedDict

import numpy as np

from mujoco_worldgen.util.types import accepts, returns
from mujoco_worldgen.parser import update_mujoco_dict
from mujoco_worldgen.transforms import closure_transform
from mujoco_worldgen.util.obj_util import get_name_index, get_axis_index
from mujoco_worldgen.util.placement import place_boxes


class Obj(object):
    '''
    The base class for the world generation. Methods that should be
    used outside are:
      - append
      - set_material
      - add_transform
      - mark
    '''

    world_params = None
    placeable = True  # Overwritten by some classes: Light, etc
    udd_callback = None  # udd_callback for object.

    def __init__(self):
        self.place_boxes = place_boxes  # allow user to override (expensive) default placement
        # ####################
        # ### First Phase ####
        # ####################
        # Create a tree of all the objects.
        # Invoked functions :
        #  - append (appends an element to the tree
        #  - mark (marks an element ; adds site in xml)
        #  - add_transform (adds posprocessing transform function).
        # Nothing has to be overrided in subclasses.

        # Determined by append
        self.children = OrderedDict()  # indexed by placement name
        # Determined by mark
        self.markers = []
        # Determined by add_transform
        self.transforms = []

        # #####################
        # ### Second Phase ####
        # #####################
        # Compute name for every element in the tree.
        # Invoked functions:
        #  - generate_name (non recursive function. Can be overriden).
        #  - to_names (recursive function)


        # ####################
        # ### Third Phase ####
        # ####################
        # Determine sizes and relative positions of objects.
        # Inoveked functions:
        #  - compile
        #  - place
        #  - generate
        self.size = None  #
        self.placements = None  # List of placements
        # Each placement is a dict {"origin": (x,y,z), "size": (x,y,z)}

        # Determined by place.
        self.relative_position = None  # (X, Y) position relative to parent

        # ###################
        # ### Four Phase ####
        # ###################
        # Determines absolute positions of generated objects.
        # Invoked functions :
        #  - set_absolute_position
        # (X, Y, Z) position in absolute (world) frame
        self.absolute_position = None

        # ####################
        # ### Fifth Phase ####
        # ####################
        # Generates xml and initial state in the simulator.
        # Invoked functions:
        #  - to_xml_dict
        #  - generate_xinit
        #  - to_xinit
        #  - generate_xml_dict

        # Checks that only allowed functions are overriden in subclasses.
        if self.__class__.__name__ != "WorldBuilder":
            for key, func in Obj.__dict__.items():
                if hasattr(func, "__call__") and key in self.__class__.__dict__:
                    assert key in ["__init__", "append", "mark", "generate_name",
                                   "generate", "generate_xinit", "generate_xml_dict"], \
                        ("Subclass %s overrides final function :%s. " % (self.__class__, key)) + \
                        "Please don't override it."
        # Extra fields
        # ============
        # Material to set on the object.
        self._material = None
        # if False object has base joints, and it's movable,
        # otherwise object is static.
        self._keep_slide_joint = [True for _ in range(3)]
        self._keep_hinge_joint = [True for _ in range(3)]

    # Removes joints from the object, and makes it not movable.
    def mark_static(self,
                    keep_slide0=False,
                    keep_slide1=False,
                    keep_slide2=False,
                    keep_hinge0=False,
                    keep_hinge1=False,
                    keep_hinge2=False):
        self._keep_slide_joint = [keep_slide0, keep_slide1, keep_slide2]
        self._keep_hinge_joint = [keep_hinge0, keep_hinge1, keep_hinge2]

    # ##################################################################
    # ########################## First Phase ###########################
    # ##################################################################
    # - create an abstract tree of objects

    @accepts(object, object, str, (tuple, list, np.ndarray))
    def append(self, obj, placement_name="top", placement_xy=None):
        '''
        Append an object to our tree.
            placement_name - name of the placement to append to
            placement_xy - ratio position of (x, y), in between 0 and 1
                this allows specification even when sizes may be unknown
                e.g. placement_xy=(0.5, 0.5) puts obj in the center
        If placement_name does not exactly match a placement, we will glob the
            end of its name and pick randomly from matched placement names.
            E.g. "shelf/shelf3" has four interior placements inner_0 to _3,
                appending to "inner" will randomly select one of those.
            Note: selection happens at generation time, so nothing is done
                during the append.
        Note: this does not check that everything fits. During the compile step
            we generate sizes and parameters, and then we can actually verify.
        '''
        assert obj.__class__.__name__ != "Material", "Material should be added with set_material."
        if not self.placeable:
            print("Don't append content to %s. It's not a valid parent." % self.__module__)
            exit(-1)
        if placement_name not in self.children:
            self.children[placement_name] = []
        if placement_xy is not None:
            assert (len(placement_xy) == 2 and
                    1. >= placement_xy[0] >= 0. and
                    1. >= placement_xy[1] >= 0.), \
                "invalid placement_xy: {}".format(placement_xy)
        self.children[placement_name].append((obj, placement_xy))
        return self

    def set_material(self, material):
        assert material.__class__.__name__ == "Material", "set_material accepts only Material objects."
        assert self._material is None, "Only one material can be specified per object."
        material.name = None
        self._material = material

    @accepts(object, str, (tuple, list, np.ndarray), (tuple, list, np.ndarray), str, (tuple, list, np.ndarray))
    def mark(self, mark_name, relative_xyz=(0.5, 0.5, 0.5), absolute_xyz=None, rgba=None,
             geom_type="sphere", size=np.array([0.1, 0.1, 0.1])):
        '''
        Similar to append(), but with markers (named sites)
            mark_name - (string) name for the mark site
            placement_name - name of the placement to use
                             common placements are "top" and "inside",
                             If None, reference base object coordinates.
            placement_xyz - ratio position of (x, y, z), in between 0 and 1
                            specifies location as fraction of full extents
                            e.g. placement_xyz=(0.5, 0.5, 0.5) is the center
                                of the volume of the placement.
            rgba - RGBA value of the mark (default is blue)
            geom_type - Geom type, like "sphere" or "box"
        Note: append() uses xy because objects always rest on the ground, but
        mark() uses xyz because sites are unaffected by gravity.
        '''
        if not self.placeable:
            print("Don't mark %s. It doesn't make sense." % self.__module__)
            exit(-1)
        assert mark_name not in [m['name'] for m in self.markers], \
            "Marker with name {} already exists".format(mark_name)
        if rgba is None:
            rgba = np.array([0., 0., 1., 1.])
        else:
            if isinstance(rgba, tuple):
                rgba = list(rgba)
            if isinstance(rgba, list):
                rgba = np.array(rgba)
            assert len(rgba.shape) == 1 and rgba.shape[0] in [3, 4], "rgba has incorrect shape"
            if rgba.shape[0] == 3:
                rgba = np.concatenate([rgba, np.ones(1)])
        rgba = rgba.astype(np.float32)
        marker = {'name': mark_name,
                  'relative_xyz': relative_xyz,
                  'absolute_xyz': absolute_xyz,
                  'rgba': rgba,
                  'type': geom_type,
                  'size': size}
        # TODO: use Maker() class of some form,
        # The old one isn't suitable, so maybe a new one derived from Obj?
        # For now a dictionary approximates it
        self.markers.append(marker)

    def add_transform(self, transform):
        '''
        Transforms are functions which are called on the XML dictionary
        produced by to_xml() before returning to the parent.  This happens
        in a recursive context, so it has access to all children, but not any
        of the parents.
        Because the XML dictionaries are mutable, the functions should modify
        the dictionary in place, and not return anything.
        The format of the dictionary matches that of xmltodict.
        '''
        assert hasattr(transform, "__call__"), \
            "Argument to add_transform should be a function"
        assert len(inspect.getargspec(transform).args) == 1, \
            "transform function should take a single argument " + \
            "of a type OrderedDict. This argument represents " + \
            "xml to be transformed."
        self.transforms.append(transform)
        return self

    # ##################################################################
    # ######################### Second Phase ###########################
    # ##################################################################

    def generate_name(self, name_indexes):
        if not hasattr(self, "name") or self.name is None:
            classname = self.__class__.__module__.split(".")[-1]
            self.name = get_name_index(name_indexes, classname)

    def to_names(self, name_indexes):
        ''' Recurse through all children and call generate_name() '''
        self.generate_name(name_indexes)
        for children in self.children.values():
            for child, _ in children:
                child.to_names(name_indexes)
        if self._material is not None:
            self._material.to_names(name_indexes)

    # #################################################################
    # ######################### Third Phase ###########################
    # #################################################################
    # Generate cannot fail.
    # - Generate has to set size attribute and placement.
    # not recursive.
    # determine random parameters of objects such as
    # - sizes
    # - XML to which they belong
    # - material
    # - lighting conditions
    def generate(self, world_params, random_state, placement_size):
        raise NotImplementedError(
            "This needs to be implemented in child classes.")

    # compile:
    # first step: outer loop - tries generate then placement
    # second step: compile children

    # Returns True if successful, returns False if max_tries did not find a
    # solution
    def compile(self, random_state, world_params):
        # Some children are in placements that need to be randomly drawn
        # Preprocess the list of children to draw placements for all of those
        for placement_name, children in list(self.children.items()):
            if placement_name in self.placements:
                continue  # Valid placement, nothing more to do
            # Select a placement from a set of candidates for each child
            candidates = self.placements.keys()
            matches = [
                pn for pn in candidates if pn.startswith(placement_name)]
            assert len(matches) > 0, (
                "No match found in {} for {}".format(placement_name, self.name))
            for child in children:  # Each gets an individual random draw
                choice = random_state.choice(matches)
                if choice not in self.children:
                    self.children[choice] = []
                self.children[choice].append(child)
            # Remove old placement
            del self.children[placement_name]
        if self._material is not None:
            self._material.generate(random_state, world_params, None)
        for placement_name in self.children.keys():
            max_tries = 10
            for _ in range(max_tries):
                placement = self.placements[placement_name]
                for child, _ in self.children[placement_name]:
                    child.size, child.placement = None, None
                    placement_size = np.array(placement['size'], dtype=np.float)
                    placement_size[0] -= 2 * world_params.placement_margin
                    placement_size[1] -= 2 * world_params.placement_margin
                    child.generate(random_state, world_params, placement_size)
                    if not child.placeable:
                        continue
                    assert child.size is not None, "missing size {}".format(
                        child)
                    assert child.placements is not None, "missing placements {}".format(
                        child)
                    child.relative_position = None
                success = self.place(placement_name, self.children[placement_name],
                                     random_state, world_params.placement_margin)
                if success:
                    for child, _ in self.children[placement_name]:
                        if not child.placeable:
                            continue
                        assert child.relative_position is not None, "{}".format(
                            child)
                    break  # next
            if not success:
                # TODO: debug level logging of _which_ placement failed
                return False  # one of our placements failed, so we failed

        for placement_name in self.children.keys():
            for child, _ in self.children[placement_name]:
                if not child.compile(random_state, world_params):
                    return False
        return True

    # Not recursive.
    def place(self, placement_name, children, random_state,
              placement_margin):
        # TODO: check height? right now we're ignoring height restrictions
        success = False

        # my amount of space is self.placements[placement_name]
        # each of my kids has size child.size.
        placement = self.placements[placement_name]
        placeable_children = [c for c in children if c[0].placeable]
        if len(placeable_children) == 0:
            return True  # Successfully placed all zero placeable children
        boxes = [{"size": c[0].size, "placement_xy": c[1]}
                 for c in placeable_children]
        width, height, _ = placement['size']
        locations = self.place_boxes(random_state, boxes, width, height,
                                     placement_margin=placement_margin)
        if locations is not None:
            for c, l in zip(placeable_children, locations):
                c[0].relative_position = l
            success = True
        return success

    # #################################################################
    # ######################### Fourth Phase ##########################
    # #################################################################
    def set_absolute_position(self, origin):
        '''
        Set absolute position of objects, recursing throught all children.
            origin - absolute position of this object's origin
        '''
        assert len(origin) == 3, "Invalid origin: {}".format(origin)
        assert len(self.relative_position) == 2, \
            "Invalid relative_position: {}".format(self.relative_position)
        self.absolute_position = np.array(origin, dtype=np.float)
        # Note relative_position is X,Y but our absolute_position is X,Y,Z
        self.absolute_position[:2] += self.relative_position
        for placement_name, children in self.children.items():
            placement = self.placements[placement_name]
            offset = self.absolute_position + placement['origin']
            for child, _ in children:
                if child.placeable:
                    child.set_absolute_position(offset)
        # Calculate positions of markers
        for marker in self.markers:
            if marker['relative_xyz'] is not None:
                relative_xyz = np.array(marker['relative_xyz'], dtype='f8')
                marker['position'] = relative_xyz * np.array(self.size, dtype=np.float)
                for i in range(3):
                    if np.abs(self.size[i]) < 1e-4:
                        marker["position"][i] = relative_xyz[i]
                marker['position'] -= self.size * 0.5
            elif marker['absolute_xyz'] is not None:
                marker['position'] = np.array(marker['absolute_xyz'], dtype='f8')
            else:
                assert False, 'Neither absolute nor relative xyz provided.'

    # ################################################################
    # ######################### Fifth Phase ##########################
    # ################################################################
    def to_xml_dict(self):
        '''
        Generates XML for this object and all of its children.
            see generate_xml() for parameter documentation
        Returns merged xml_dict
        '''
        full_xml_dict = OrderedDict()
        # First add all of our own xml
        self.xml_dict = self.generate_xml_dict()

        # Removed joints marked to be static. We set positions in XML instead of using qpos.
        for body in self.xml_dict.get("worldbody", {}).get("body", []):
            remaining_joints = []
            for jnt in body.get("joint", []):
                if jnt["@type"] == "slide":
                    axis_idx = get_axis_index(jnt["@axis"])
                    if self._keep_slide_joint[axis_idx]:
                        remaining_joints.append(jnt)
                    else:
                        body["@pos"][axis_idx] = float(body["@pos"][axis_idx]) + self.absolute_position[axis_idx]
                elif jnt["@type"] == "hinge":
                    axis_idx = get_axis_index(jnt["@axis"])
                    if self._keep_hinge_joint[axis_idx]:
                        remaining_joints.append(jnt)
                elif jnt["@type"] == "ball":
                    remaining_joints.append(jnt)
            body["joint"] = remaining_joints

        if len(self.markers) > 0:
            bodies = [body for body in self.xml_dict["worldbody"]
                      ["body"] if "annotation" not in body["@name"] and
                                  ("@mocap" not in body or not body["@mocap"])]
            assert len(bodies) == 1, ("Object %s should have only one body " % self) + \
                "to attach markers to. Otherwise mark() is" + \
                "ambiguous."
            body = bodies[0]
            if "site" not in body:
                body["site"] = []
            for marker in self.markers:
                site = OrderedDict()
                site['@name'] = marker['name']
                site['@pos'] = marker['position']
                site['@size'] = marker['size']
                site['@rgba'] = marker['rgba']
                site['@type'] = marker['type']
                body['site'].append(site)

        # Adding material influences nodes of the parent.
        if self._material is not None:
            update_mujoco_dict(self.xml_dict, self._material.generate_xml_dict())

            def assign_material(node):
                if "geom" in node:
                    for g in node["geom"]:
                        g["@material"] = self._material.name
            closure_transform(assign_material)(self.xml_dict)

        update_mujoco_dict(full_xml_dict, self.xml_dict)
        for transform in self.transforms:
            transform(full_xml_dict)
        # Then add the xml of all of our children
        for children in self.children.values():
            for child, _ in children:
                child_dict = child.to_xml_dict()
                update_mujoco_dict(full_xml_dict, child_dict)

        return full_xml_dict

    def generate_xinit(self):
        # MuJoCo uses center of geom as origin, while we use bottom corner
        if not self.placeable:
            return {}
        position = self.absolute_position
        position_xinit = {}
        # extracts names of three top level slide joints from
        # the body.
        for body in self.xml_dict["worldbody"]["body"]:
            for jnt in body.get("joint", []):
                if jnt["@type"] == "slide":
                    idx = get_axis_index(jnt["@axis"])
                    position_xinit[jnt["@name"]] = position[idx]
        # Some people add transforms which remove joints.
        # Only generate xinit terms for joints that remain
        xinit = {}
        for body in self.xml_dict["worldbody"]["body"]:
            for joint in body.get('joint', []):
                joint_name = joint.get('@name', '')
                if joint_name in position_xinit:
                    xinit[joint_name] = position_xinit[joint_name]

        if hasattr(self, "default_qpos") and self.default_qpos is not None:
            for joint, value in self.default_qpos.items():
                if not joint.startswith(self.name + ':'):
                    joint = self.name + ":" + joint
                xinit[joint] = value
        return xinit

    def to_xinit(self):
        '''
        Recurse through all children and return merged xinit dictionary.
            See generate_xinit() for more info.
        '''
        xinit = self.generate_xinit()
        for children in self.children.values():
            for child, _ in children:
                xinit.update(child.to_xinit())
        return xinit

    def to_udd_callback(self):
        '''
        Recurse through all children and return merged udd_callback.
        '''
        udd_callbacks = []
        if self.udd_callback is not None:
            udd_callbacks.append(self.udd_callback)
        for children in self.children.values():
            for child, _ in children:
                udd_callbacks += child.to_udd_callback()
        return udd_callbacks

    @returns(OrderedDict)
    def generate_xml_dict(self):
        '''
        Generate XML DOM nodes needed for MuJoCo model.
            doc - XML Document, used to create elements/nodes
            name_indexes - dictionary to keep track of names,
                see obj_util.get_name_index() for internals
        Returns a dictionary with keys as names of top-level nodes:
            e.g. 'worldbody', 'materials', 'assets'
        And the values are lists of XML DOM nodes
        '''
        raise NotImplementedError('Implement in subclasses!')

    def __repr__(self):
        outer = str(self.__class__.__name__)
        inner = []
        if self.children is not None and len(self.children) > 0:
            inner.append('children={}'.format(self.children))
        if self.relative_position is not None:
            inner.append('relpos={}'.format(self.relative_position))
        if self.absolute_position is not None:
            inner.append('abspos={}'.format(self.absolute_position))
        return '{}({})'.format(outer, ', '.join(inner))
