__author__ = 'ivan'

import numpy as np

from primitives import Ray

REFLECTION_COEF = 0.5


def calculate_object_color(hit):
    if hit is None:
        return np.array([0, 0, 0])

    return hit.object.color


def calculate_normal(hit):
    if hit is None:
        return np.array([0, 0, 0])

    return hit.normal * 0.5 + 0.5


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

    def _trace_shadow_ray(self, ray, tol):
        """
        :param ray: ray to cast
        :return: hit, if ray intersects with some object, None otherwise
        """
        for obj in self.scene.objects:
            hit = obj.intersect(ray, tol)
            if hit is not None:
                return hit

        return None

    def _trace_ray(self, ray, calc_color, tol, tracing_depth):
        """
        :param ray: object of type Ray
        :param calc_color: function, which takes Hit object as parameter and returns color of object in this point
        :return: result of execution calc_color function
        """

        min_dist = 1e8
        min_hit = None

        for obj in self.scene.objects:
            hit = obj.intersect(ray, tol)
            if hit is None:
                continue

            dist = hit.distance()
            if dist < min_dist:
                min_dist = dist
                min_hit = hit

        obj_color = calc_color(min_hit)

        res_shading = np.array([0.0, 0.0, 0.0])

        if min_hit is not None:
            # Tracing shadow rays
            for light in self.scene.lights:
                shadow_ray = Ray(min_hit.point, light.origin - min_hit.point)
                shadow_hit = self._trace_shadow_ray(shadow_ray, tol)

                if shadow_hit is None:
                    res_shading += light.get_intensity(shadow_ray) * obj_color * max(0, np.dot(min_hit.normal, shadow_ray.direction))

            # Tracing reflection ray
            if tracing_depth > 0:

                reflection_ray_dir = ray.direction - 2.0 * (np.dot(ray.direction, min_hit.normal)) * min_hit.normal
                reflection_ray = Ray(min_hit.point, reflection_ray_dir)
                reflection_shading = self._trace_ray(reflection_ray, calc_color, tol, tracing_depth - 1)

                res_shading += REFLECTION_COEF * reflection_shading

        return res_shading

    def run(self, calc_result, dim_per_color=1, tol=1e-3, tracing_depth=1):
        result = None
        if dim_per_color > 1:
            result = np.zeros((self.camera.resolution_y, self.camera.resolution_x, dim_per_color))
        else:
            result = np.zeros((self.camera.resolution_y, self.camera.resolution_x))

        for y in range(self.camera.resolution_y):
            for x in range(self.camera.resolution_x):
                point = (x, y)
                normalized_point = self.camera.to_normalized(point)
                ray = self.camera.get_ray(normalized_point)
                result[y, x] = self._trace_ray(ray, calc_result, tol, tracing_depth)

        result *= (1.0 / np.max(result))
        return result
