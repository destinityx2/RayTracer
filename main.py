__author__ = 'ivan'

import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from parser import parse_camera_scene
from ray_tracer import RayTracer, calculate_distance, calculate_normal, calculate_object_color

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--scene', help='path to the scene', action='store', dest='scene', required=True)
arg_parser.add_argument('--output', action='store', dest='output', required=True)
arg_parser.add_argument('--resolution_x', action='store', dest='resolution_x', type=int, default=100)
arg_parser.add_argument('--resolution_y', action='store', dest='resolution_y', type=int, default=100)
arg_parser.add_argument('--trace_depth', action='store', dest='trace_depth', type=int, default=0)
arg_parser.add_argument('--normal_as_color', action='store_true', default=False)
arg_parser.add_argument('--distance_as_color', action='store_true', default=False)
arg_parser.add_argument('--dist_range', action='store', dest='dist_range', type=int)

args = arg_parser.parse_args()

camera, scene = parse_camera_scene(args.scene)

tracer = RayTracer(scene, camera)
camera.resolution_x = args.resolution_x
camera.resolution_y = args.resolution_y

if args.normal_as_color:
    res_img = tracer.run(calculate_normal, 3)
    plt.imsave(args.output, res_img)
elif args.distance_as_color:
    calc_dist = lambda hit: calculate_distance(hit, args.dist_range)
    res_img = tracer.run(calc_dist)
    plt.imsave(args.output, res_img, cmap=cm.gray)
else:
    res_img = tracer.run(calculate_object_color, 3, camera.near_clip, tracing_depth=args.trace_depth)
    plt.imsave(args.output, res_img)