__author__ = 'ivan'

import numpy as np
from parser import parse_camera_scene
from scene import Scene
from objects import Sphere, Plane, Triangle
from ray_tracer import RayTracer, calculate_distance, calculate_normal, calculate_object_color
from camera import Camera
from lights import PointLight

import matplotlib.pyplot as plt
import matplotlib.cm as cm


camera, scene = parse_camera_scene('scene_axes_test.yml')

# obj1 = Sphere(0.3, np.array((0, 0, 2)))
# obj2 = Sphere(0.3, np.array((0, 1, 2)))
# obj3 = Plane(np.array((0, 0, 4)), np.array((0, 0, 1)))
# obj4 = Triangle(np.array((0, 1, 3)), n    p.array((1, -1, 3)), np.array((3, -2, 3)))
# objects = [obj1, obj2, obj3, obj4]
#
# light1 = PointLight(np.array((0, 1, -1)), 0.5)
#
# lights = [light1]
#
# scene = Scene(objects, lights)
#
# camera = Camera(np.array((0, 0, 0)), np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))), (45, 45), 200, 200)

tracer = RayTracer(scene, camera)

# calc_dist = lambda hit: calculate_distance(hit, 10)
# res_img = tracer.run(calc_dist)
# plt.imsave('result.png', res_img, cmap=cm.gray)

# res_img = tracer.run(calculate_normal, 3)
# plt.imsave('result.png', res_img)

res_img = tracer.run(calculate_object_color, 3)
plt.imsave('result.png', res_img)