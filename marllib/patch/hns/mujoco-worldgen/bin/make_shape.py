#!/usr/bin/env python
# Make STL files for simple geometric shapes
# This file is expected to be edited and ran, or used as a module
# It's a bit of a hack, but it makes objects that work!
import os
import numpy as np
from collections import OrderedDict
from itertools import chain, combinations, product
import xmltodict
from pyhull.convex_hull import ConvexHull
from stl.mesh import Mesh
from mujoco_worldgen.util.path import worldgen_path

# Basic (half) unit length we normalize to
u = 0.038  # 38 millimeters in meters


def norm(shape):
    ''' Center a shape over the origin, and scale it to fit in unit box '''
    mins, maxs = np.min(shape, axis=0), np.max(shape, axis=0)
    return (shape - (maxs + mins) / 2) / (maxs - mins) * u * 2


def roll3(points):
    ''' Return a set of rotated 3d points (used for construction) '''
    return np.vstack([np.roll(points, i) for i in range(3)])


def subdivide(shape):
    ''' Take a triangulated sphere and subdivide each face. '''
    # https://medium.com/game-dev-daily/d7956b825db4 - Icosahedron section
    hull = ConvexHull(shape)
    radius = np.mean(np.linalg.norm(hull.points, axis=1))  # Estimate radius
    edges = set(chain(*[combinations(v, 2) for v in hull.vertices]))
    midpoints = np.mean(hull.points.take(list(edges), axis=0), axis=1)
    newpoints = midpoints / np.linalg.norm(midpoints, axis=1)[:, None] * radius
    return norm(np.vstack((hull.points, newpoints)))


def top(shape):
    ''' Get only the top half (z >= 0) points in a shape. '''
    return shape[np.where(shape.T[2] >= 0)]


phi = (1 + 5 ** .5) / 2  # Golden ratio
ihp = 1 / phi  # Inverted golden ratio (phi backwards)

# Construct tetrahedron from unit axes and projected (ones) point
tetra = norm(np.vstack((np.ones(3), np.eye(3))))
# Construct cube from tetrahedron and inverted tetrahedron
cube = np.vstack((tetra, -tetra))
# Construct octahedron from unit axes and inverted unit axes
octa = norm(np.vstack((np.eye(3), -np.eye(3))))
# Construct icosahedron from (phi, 1) planes
ico_plane = np.array(list(product([1, -1], [phi, -phi], [0])))
icosa = norm(roll3(ico_plane))
# Construct dodecahedron from unit cube and (phi, ihp) planes
dod_cube = np.array(list(product(*([(-1, 1)] * 3))))
dod_plane = np.array(list(product([ihp, -ihp], [phi, -phi], [0])))
dodeca = norm(np.vstack((dod_cube, roll3(dod_plane))))
# Subdivided icosahedrons
sphere80 = subdivide(icosa)  # pentakis icosidodecahedron
sphere320 = subdivide(sphere80)  # 320 sided spherical polyhedra
sphere1280 = subdivide(sphere320)  # 1280 sided spherical polyhedra

# Josh's shapes (basic length 38mm)
line = np.linspace(0, 2 * np.pi, 100)
circle = np.c_[np.cos(line), np.sin(line), np.zeros(100)] * u
halfsphere = np.r_[circle, top(sphere1280)]
cone = np.r_[circle, np.array([[0, 0, 2 * u]])]
h = u * .75 ** .5  # half-height of the hexagon
halfagon = np.array([[u, 0, 0], [u / 2, h, 0], [-u / 2, h, 0]])
hexagon = np.r_[halfagon, -halfagon]
hexprism = np.r_[hexagon, hexagon + np.array([[0, 0, 2 * u]])]
triangle = np.array([[u, 0, 0], [-u, 0, 0], [0, 2 * h, 0]])
tetra = np.r_[triangle, np.array([[0, h, 2 * u]])]
triprism = np.r_[triangle, triangle + np.array([[0, 0, 2 * u]])]
square = np.array([[u, u, 0], [-u, u, 0], [-u, -u, 0], [u, -u, 0]])
pyramid = np.r_[square, np.array([[0, 0, u * 2]])]


def build_stl(name, points):
    ''' Given a set point points, make a STL file of the convex hull. '''
    points = np.array(points)
    points -= np.min(points, axis=0)  # Move bound to origin
    hull = ConvexHull(points)
    shape = Mesh(np.zeros(len(hull.vertices), dtype=Mesh.dtype))
    for i, vertex in enumerate(hull.vertices):
        shape.vectors[i] = hull.points[vertex][::-1]  # Turn it inside out
    size = np.max(hull.points, axis=0)
    return shape, size


def build_xml(name, size):
    ''' Make the corresponding XML file to match the STL file '''
    path = os.path.join(os.path.join('shapes', name), name + '.stl')
    mesh = OrderedDict([('@name', name), ('@file', path), ('@scale', '1 1 1')])
    asset = OrderedDict(mesh=mesh)

    rgba = ' '.join(str(v) for v in np.random.uniform(size=3)) + ' 1'
    halfsize = ' '.join([str(v) for v in size * 0.5])

    joints = []
    for joint_type in ('slide', 'hinge'):
        for i, axis in enumerate(('0 0 1', '0 1 0', '1 0 0')):
            joints.append(OrderedDict([('@name', '%s%d' % (joint_type, i)),
                                       ('@type', joint_type),
                                       ('@pos', '0 0 0'),
                                       ('@axis', axis),
                                       ('@damping', '10')]))

    body_geom = OrderedDict([('@name', name),
                             ('@pos', '0 0 0'),
                             ('@type', 'mesh'),
                             ('@mesh', name),
                             ('@rgba', rgba)])
    body = OrderedDict([('@name', name),
                        ('@pos', '0 0 0'),
                        ('geom', body_geom),
                        ('joint', joints)])

    outer_geom = OrderedDict([('@type', 'box'),
                              ('@pos', '0 0 0'),
                              ('@size', halfsize)])
    outer_bound = OrderedDict([('@name', 'annotation:outer_bound'),
                               ('@pos', halfsize),
                               ('geom', outer_geom)])
    worldbody = OrderedDict(body=[body, outer_bound])
    return OrderedDict(mujoco=OrderedDict(asset=asset, worldbody=worldbody))


def make_shape(name, points):
    ''' Make the STL and XML, and save both to the proper directories. '''
    # Make the STL and XML
    shape, size = build_stl(name, points)
    xml_dict = build_xml(name, size)
    # Make the directory to save files to if we have to
    xml_dirname = worldgen_path('assets', 'xmls', 'shapes', name)
    stl_dirname = worldgen_path('assets', 'stls', 'shapes', name)
    os.makedirs(xml_dirname, exist_ok=True)
    os.makedirs(stl_dirname, exist_ok=True)
    # Save the STL and XML to our new directories
    shape.save(os.path.join(stl_dirname, name + '.stl'))
    with open(os.path.join(xml_dirname, 'main.xml'), 'w') as f:
        f.write(xmltodict.unparse(xml_dict, pretty=True))


shapes_to_build = {'cube': cube,
                   'octa': octa,
                   'icosa': icosa,
                   'dodeca': dodeca,
                   'sphere80': sphere80,
                   'sphere320': sphere320,
                   'sphere1280': sphere1280,
                   'halfsphere': halfsphere,
                   'cone': cone,
                   'hexprism': hexprism,
                   'tetra': tetra,
                   'triprism': triprism,
                   'pyramid': pyramid}

test_shapes_to_build = {'tetra': tetra,
                        'triprism': triprism,
                        'pyramid': pyramid}


if __name__ == '__main__':
    for name, points in test_shapes_to_build.items():
        make_shape(name, points)
