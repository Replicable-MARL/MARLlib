import numpy as np
from collections import OrderedDict
# TODO: Write more comments.

from mujoco_worldgen.util.types import accepts, returns, maybe


def get_camera_xyaxes(camera_pos, target_pos):
    '''
    Calculate the "xyaxis" frame orientation for a camera,
        given the camera position and target position.
    Returns a 6-vector of the x axis and y axis of the camera frame.
        See http://www.mujoco.org/book/modeling.html#COrientation "xyaxes"
    '''
    camera_pos = np.array(camera_pos)
    target_pos = np.array(target_pos)
    assert camera_pos.shape == (
        3,), "Bad camera position {}".format(camera_pos)
    assert target_pos.shape == (
        3,), "Bad target position {}".format(target_pos)
    vector = target_pos - camera_pos
    cross = np.cross(vector, np.array([0, 0, 1]))
    cross2 = np.cross(cross, vector)
    return np.concatenate((cross, cross2))


@accepts(OrderedDict, str, maybe(np.ndarray))
def add_annotation_bound(xml_dict, annotation_name, bound):
    '''
    Add an annotation bounding box to and XML dictionary.
    Annotation name will be "annotation:" + annotation_name
    Bound is given as a 2 x 3 np.ndarray, and represents:
        [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    '''
    if bound is None:
        return xml_dict  # Nothing to do here
    assert bound.shape == (2, 3), "Bound must be 2 x 3 (see docstring)."
    assert 'worldbody' in xml_dict, "XML must have worldbody"
    worldbody = xml_dict['worldbody']
    assert 'body' in worldbody, "XML worldbody must have bodies"
    name = 'annotation:' + annotation_name
    # Remove old annotations with the same name before inserting new annotation
    bodies = [body for body in worldbody['body'] if body.get('@name') != name]
    rgba = np.random.uniform(size=4)
    rgba[3] = 0.1  # annotation is almost transparent.
    geom = OrderedDict([('@conaffinity', 0),
                        ('@contype', 0),
                        ('@mass', 0.0),
                        ('@pos', np.zeros(3)),
                        ('@rgba', rgba),
                        ('@size', (bound[1] - bound[0]) / 2),  # halfsize
                        ('@type', 'box')])
    annotation = OrderedDict([('@name', name),
                              ('@pos', bound.mean(axis=0)),  # center pos
                              ('geom', [geom])])
    bodies.append(annotation)
    worldbody['body'] = bodies
    print('adding annotation bound (size)', name, bound[1] - bound[0])


@accepts(OrderedDict)
@returns(OrderedDict)
def get_xml_meshes(xml_dict):
    ''' Get dictionary of all the mesh names -> filenames in a parsed XML. '''
    meshes = OrderedDict()
    for mesh in xml_dict.get('asset', {}).get('mesh', []):
        assert '@name' in mesh, "Mesh missing name: {}".format(mesh)
        assert '@file' in mesh, "Mesh missing file: {}".format(mesh)
        scale = np.ones(3)
        if "@scale" in mesh:
            scale = mesh["@scale"]
        meshes[mesh['@name']] = (mesh['@file'], scale)
    return meshes


@accepts(OrderedDict, str)
def recursive_rename(xml_dict, prefix):
    attrs = ["@name", "@joint", "@jointinparent", "@class", "@source",
             "@target", "@childclass", "@body1", "@body2", "@mesh",
             "@joint1", "@joint2", "@geom", "@geom1", "@geom2", "@site",
             "@material", "@texture", "@tendon", "@sidesite", "@actuator"]
    names = ["geom", "joint", "jointinparent", "body", "motor", "freejoint", "general",
             "position", "default", "weld", "exclude", "mesh",
             "site", "pair", "jointpos", "touch", "texture", "material",
             "fixed", "spatial", "motor", "actuatorfrc"]
    if not isinstance(xml_dict, OrderedDict):
        return
    for key in list(xml_dict.keys()):
        value_dict = xml_dict[key]
        if isinstance(value_dict, OrderedDict):
            value_dict = [value_dict]
        if key in names:
            assert isinstance(
                value_dict, list), "Invalid type for value {}".format(value_dict)
            for value in value_dict:
                for attr in list(value.keys()):
                    if attr in attrs:
                        if not value[attr].startswith(prefix + ':'):
                            value[attr] = prefix + ':' + value[attr]
        if isinstance(value_dict, list):
            for value in value_dict:
                recursive_rename(value, prefix)


def establish_size(min_size, max_size):
    if isinstance(min_size, (float, int)):
        min_size = np.ones(3) * float(min_size)
    if isinstance(max_size, (float, int)):
        max_size = np.ones(3) * float(max_size)
    if max_size is None and min_size is not None:
        max_size = min_size
    if max_size is None and min_size is None:
        min_size = np.ones(3) * 0.1
        max_size = np.ones(3) * 0.1
    if isinstance(min_size, (list, tuple)):
        min_size = np.array(min_size, dtype=np.float64)
    if isinstance(max_size, (list, tuple)):
        max_size = np.array(max_size, dtype=np.float64)
    assert(isinstance(min_size[0], float))
    assert(isinstance(max_size[0], float))
    for i in range(3):
        assert(max_size[i] >= min_size[i])
    return min_size, max_size


def get_name_index(name_indexes, name):
    '''
    Update the name index and return new name
    name - name to look up index for, e.g. "geom"
    name_indexes - dictionary to keep track of names, e.g.
        {'geom': 4} means there are 4 geom objects, and the next
        geom object should be called "geom4" and the dictionary updated
        to be {'geom': 5}
    Returns name with index attached, e.g. "geom4"
    '''
    if name not in name_indexes:
        name_indexes[name] = 0
    result = '%s%d' % (name, name_indexes[name])
    name_indexes[name] += 1
    return result


def get_body_xml_node(name, use_joints=False):
    '''
    Build a body XML dict for use in object models.
        name - name for the body (should be unique in the model, e.g. "geom4")
        joints - if True, add 6 degrees of freedom joints (slide, hinge)
    Returns named XML body node.
    '''
    body = OrderedDict()
    body['@name'] = name
    body['@pos'] = np.zeros(3)

    if use_joints:
        joints = []
        for axis_type in ('slide', 'hinge'):
            for i, axis in enumerate(np.eye(3)):
                joint = OrderedDict()
                joint['@name'] = "%s:%s%d" % (name, axis_type, i)
                joint['@axis'] = axis
                joint['@type'] = axis_type
                joint['@damping'] = 0.01
                joint['@pos'] = np.zeros(3)
                joints.append(joint)
        body['joint'] = joints
    return body


@accepts(np.ndarray)
@returns(int)
def get_axis_index(axis):
    '''
    Returns axis index from a string:
    # return 0 for axis = 1 0 0
    # return 1 for axis = 0 1 0
    # return 2 for axis = 0 0 1
    '''
    assert axis.shape[0] == 3
    for i in range(3):
        if axis[i] != 0:
            return i
    assert False, "axis should be of a form (1 0 0), or (0 1 0), or (0 0 1)." \
                  "Current axis = %s, it's not. Failing." % str(axis)
