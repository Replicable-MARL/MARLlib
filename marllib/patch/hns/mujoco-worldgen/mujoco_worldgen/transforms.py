from collections import OrderedDict

'''
    Transforms are functions which modify the world in-place.
    They can be used to add, remove or change specific attributes, tags, etc.
'''

def closure_transform(closure):
    '''
        Call closure on every OrderedDict.
        This transform is usually not used directly, it is just called internally
        by other transforms.
    '''
    def recursion(xml_dict):
        closure(xml_dict)
        for key in list(xml_dict.keys()):
            values = xml_dict[key]
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if isinstance(value, OrderedDict):
                    recursion(value)
    return recursion


def set_geom_attr_transform(name, value):
    ''' Sets an attribute to a specific value on all geoms '''
    return set_node_attr_transform('geom', name, value)


def set_node_attr_transform(nodename, attrname, value):
    '''
        Sets an attribute to a specific value on every node of the specified type (e.g. geoms).
    '''
    def fun(xml_dict):
        def closure(node):
            if nodename in node:
                for child in node[nodename]:
                    child["@" + attrname] = value
        return closure_transform(closure)(xml_dict)
    return fun
