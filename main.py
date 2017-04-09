__author__ = 'ivan'

import numpy as np
from yaml import load_all
from scene import Scene
from objects import Sphere, Plane, Triangle
from ray_tracer import RayTracer, calculate_distance, calculate_normal
from camera import Camera

import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open('scene.yml') as f:
    text = ''.join(f.readlines())
    print(text)
    scene = load_all(text)
    for x in scene:
        print(x)


obj1 = Sphere(2, np.array((1, 1, 6)))
obj2 = Sphere(0.5, np.array((-1, -1, 4)))
obj3 = Plane(np.array((0, 0, 4.5)), np.array((0, 0, 1)))
obj4 = Triangle(np.array((0, 1, 3)), np.array((1, -1, 3)), np.array((3, -2, 3)))
objects = [obj1, obj2, obj3, obj4]

scene = Scene(objects, [])

camera = Camera(np.array((0, 0, 0)), np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1))), 45, 200, 200)

tracer = RayTracer(scene, camera)

calc_dist = lambda hit: calculate_distance(hit, 5)
res_img = tracer.run(calc_dist)
plt.imsave('result.bmp', res_img, cmap=cm.gray)

# res_img = tracer.run(calculate_normal, 3)
# plt.imsave('result.bmp', res_img)