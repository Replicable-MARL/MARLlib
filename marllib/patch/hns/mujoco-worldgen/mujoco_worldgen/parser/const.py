# Automatically generated. Do not modify!
'''
We represent XML as OrderedDict and lists.
Some nodes have OrderedDict value (mostly top-lever such as worldbody.
some nodes have list values (mostly lower level). However,
MuJoCo is not consistent with it. We automatically checked in many XMLs
if given values occur once (then, they are converted to OrderedDict) or
multiple times (then, they are converted to list). script get_const.py
determines it, and writes result to const.py. Value in variable list_types
describes nodes that of a type list.
Moreover, same arguments are indeed floats, but given XML might store them
as ints, e.g. pos = "1 1 1". Here we determine actuall datatype.
'''


list_types = set([
         "actuatorfrc", 
         "body", 
         "default", 
         "exclude", 
         "fixed", 
         "geom", 
         "include", 
         "joint", 
         "jointpos", 
         "light", 
         "material", 
         "mesh", 
         "motor",
         "general",
         "pair", 
         "position", 
         "site", 
         "texture", 
         "touch", 
         "weld", 
])

float_arg_types = set([
         "@armature", 
         "@axis", 
         "@axisangle", 
         "@coef", 
         "@ctrlrange", 
         "@damping", 
         "@density", 
         "@diaginertia", 
         "@diffuse", 
         "@dir", 
         "@euler", 
         "@force", 
         "@forcerange", 
         "@fovy", 
         "@friction", 
         "@frictionloss", 
         "@fromto", 
         "@gear", 
         "@kp", 
         "@margin", 
         "@markrgb", 
         "@mass", 
         "@polycoef", 
         "@pos", 
         "@quat", 
         "@random", 
         "@range", 
         "@ref", 
         "@reflectance", 
         "@rgb1", 
         "@rgb2", 
         "@rgba", 
         "@scale", 
         "@shininess", 
         "@size", 
         "@solimp", 
         "@solimplimit", 
         "@solref", 
         "@solreflimit", 
         "@specular", 
         "@stiffness", 
         "@timestep", 
         "@transformation", 
])
