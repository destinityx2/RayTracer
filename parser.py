__author__ = 'ivan'

from yaml import load_all
from camera import Camera
from objects import Sphere, Triangle, Plane
from scene import Scene
from lights import PointLight
import numpy as np

angle_to_radian = np.pi / 180.0

DEFAULT_X_RESOLUTION = 100
DEFAULT_Y_RESOLUTION = 100


def build_transformation_matrix(alpha, beta, gamma, x, y, z):
    # http://planning.cs.uiuc.edu/node102.html
    from numpy import sin as s
    from numpy import cos as c
    R = np.zeros((4, 4))
    # XYZ transformation
    R[0, 3] = x
    R[1, 3] = y
    R[2, 3] = z

    # Angles transformation
    R[0, 0] = c(alpha) * c(beta)
    R[0, 1] = c(alpha) * s(beta) * s(gamma) - s(alpha) * c(gamma)
    R[0, 2] = c(alpha) * s(beta) * c(gamma) + s(alpha) * s(gamma)
    R[1, 0] = s(alpha) * c(beta)
    R[1, 1] = s(alpha) * s(beta) * s(gamma) + c(alpha) * c(gamma)
    R[1, 2] = s(alpha) * s(beta) * c(gamma) - c(alpha) * s(gamma)
    R[2, 0] = -s(beta)
    R[2, 1] = c(beta) * s(gamma)
    R[2, 2] = c(beta) * c(gamma)

    # Additional 1 in [3, 3]
    R[3, 3] = 1

    return R


def lcs_to_transformation_matrix(lcs):
    x, y, z = lcs['x'], lcs['y'], lcs['z']
    h, p, r = lcs['h'], lcs['p'], lcs['r']
    return build_transformation_matrix(h, p, r, x, y, z)


def get_object(node_yml):
    if 'sphere' in node_yml:
        sphere_yml = node_yml['sphere']
        return Sphere(sphere_yml['r'], np.array([0, 0, 0]))
    elif 'plane' in node_yml:
        plane_yml = node_yml['plane']
        normal = np.array([plane_yml['normal_x'], plane_yml['normal_y'], plane_yml['normal_z']])
        return Plane(np.array([0, 0, 0]), normal)
    elif 'triangle' in node_yml:
        triangle_yml = node_yml['triangle']
        p0 = np.array([triangle_yml['x0'], triangle_yml['y0'], triangle_yml['z0']])
        p1 = np.array([triangle_yml['x1'], triangle_yml['y1'], triangle_yml['z1']])
        p2 = np.array([triangle_yml['x2'], triangle_yml['y2'], triangle_yml['z2']])
        return Triangle(p0, p1, p2)

    raise RuntimeError("Found not supported type of object")


def transform_object(obj, transform_matrix):
    def transform_3d_point(point):
        p = []
        p.extend(point)
        p.append(1)
        new_point = transform_matrix.dot(np.array(p))
        return new_point[:3]

    def transform_3d_vector(vector):
        v = []
        v.extend(vector)
        v.append(0)
        new_vector = transform_matrix.dot(np.array(v))
        return new_vector[:3]

    if isinstance(obj, Sphere):
        obj.origin = transform_3d_point(obj.origin)
        return obj
    elif isinstance(obj, Triangle):
        points = [obj.p0, obj.p1, obj.p2]
        new_points = [transform_3d_point(p) for p in points]
        obj.p0 = new_points[0]
        obj.p1 = new_points[1]
        obj.p2 = new_points[2]
        return obj
    elif isinstance(obj, Plane):
        obj.origin = transform_3d_point(obj.origin)
        obj.normal = transform_3d_vector(obj.normal)
        return obj


def _explore_node(node_yml):
    objs = []

    transform_matrix = lcs_to_transformation_matrix(node_yml[0]['lcs'])
    cur_object = get_object(node_yml[1])
    color = node_yml[2]['material']['color']
    cur_object.color = np.array([color['r'], color['g'], color['b']])
    objs.append(cur_object)

    for i in range(3, len(node_yml)):
        for inner_obj in _explore_node(node_yml[i]['node']):
            objs.append(inner_obj)

    result_objs = []
    for x in objs:
        result_objs.append(transform_object(x, transform_matrix))

    return result_objs


def _parse_lights(lights_yml):
    res_lights = []

    for light in lights_yml:
        if 'pointLight' not in light:
            raise RuntimeError("Unsupported type of light")

        point_light = light['pointLight']
        origin = point_light['x'], point_light['y'], point_light['z']
        intensity = point_light['intensity']
        res_lights.append(PointLight(origin, intensity))

    return res_lights


def _parse_scene(scene_yml, lights_yml):
    res_objects = []
    for node in scene_yml:
        node_objs = _explore_node(node['node'])
        res_objects.extend(node_objs)

    lights = _parse_lights(lights_yml)
    return Scene(res_objects, lights)


def _parse_camera(camera_yml):
    pos = camera_yml['position']
    orientation = camera_yml['orientation']
    fov_x = camera_yml['fov_x']
    fov_y = camera_yml['fov_y']
    near_clip = camera_yml['near_clip']

    x, y, z = pos['x'], pos['y'], pos['z']

    h, p, r = orientation['h'] * angle_to_radian, orientation['p'] * angle_to_radian, orientation['r'] * angle_to_radian
    tr_matr = build_transformation_matrix(h, p, r, 0, 0, 0)
    u = tr_matr.dot(np.array([1, 0, 0, 0]))[:3]
    v = tr_matr.dot(np.array([0, 1, 0, 0]))[:3]
    w = tr_matr.dot(np.array([0, 0, 1, 0]))[:3]

    return Camera(np.array((x, y, z)), np.array((u, v, w)), (fov_x, fov_y), DEFAULT_X_RESOLUTION, DEFAULT_Y_RESOLUTION, near_clip)


def parse_camera_scene(filepath):
    """
    :param filepath: Path to the ,yml file
    :return: tuple (camera, scene), where camera is the object of type Camera and scene is of type Scene
    """
    with open(filepath) as f:
        text = ''.join(f.readlines())
        description = next(load_all(text))
        if 'scene' not in description:
            raise RuntimeError("No 'scene' object in file {}".format(filepath))

        if 'camera' not in description:
            raise RuntimeError("No 'camera' object in file {}".format(filepath))

        if 'lights' not in description:
            raise RuntimeError("No 'lights' object in file {}".format(filepath))

        scene_yml = description['scene']
        camera_yml = description['camera']
        lights_yml = description['lights']

        return _parse_camera(camera_yml), _parse_scene(scene_yml, lights_yml)
