# RayTracer
Implementation of the Ray Tracing algorithm in Python.

The program has the following syntax:

usage: main.py [-h] --scene SCENE --output OUTPUT
               [--resolution_x RESOLUTION_X] [--resolution_y RESOLUTION_Y]
               [--trace_depth TRACE_DEPTH] [--normal_as_color]
               [--distance_as_color] [--dist_range DIST_RANGE]

Example of using:
python main.py --scene=scene_2.yml --output=result.png --resolution_x=500 --resolution_y=500 --trace_depth=1
