__author__ = 'ivan'

import numpy as np


def calculate_normal(hit):
    if hit is None:
        return np.array([0, 0, 0])

    return np.abs(hit.normal)


def calculate_distance(hit, dist_range):
    if hit is None:
        return 0

    dist = hit.distance()

    dist = min(dist, dist_range)
    dist /= dist_range

    return 1 - dist


class RayTracer:
    def __init__(self, scene, camera):
        self.scene = scene
        self.camera = camera

    def _trace_ray(self, point, calc_result):
        """
        :param point: point on the screen of type tuple (x: int, y: int)
        :param calc_result: function, which takes Hit object as parameter
        :return: result of execution calc_result function
        """

        normalized_point = self.camera.to_normalized(point)
        ray = self.camera.get_ray(normalized_point)

        min_dist = 1e8
        min_hit = None

        for obj in self.scene.objects:
            hit = obj.intersect(ray)
            if hit is None:
                continue

            dist = hit.distance()
            if dist < min_dist:
                min_dist = dist
                min_hit = hit

        return calc_result(min_hit)

    def run(self, calc_result, dim_per_color=1):
        result = None
        if dim_per_color > 1:
            result = np.zeros((self.camera.resolution_y, self.camera.resolution_x, dim_per_color))
        else:
            result = np.zeros((self.camera.resolution_y, self.camera.resolution_x))

        for y in range(self.camera.resolution_y):
            for x in range(self.camera.resolution_x):
                point = (x, y)
                result[y, x] = self._trace_ray(point, calc_result)

        return result
