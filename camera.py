__author__ = 'ivan'

import numpy as np
from primitives import Ray


class Camera:
    def __init__(self, origin, direction, fov_angle, resolution_x, resolution_y):
        self.origin = origin
        self.direction = direction
        self.fov_angle = fov_angle
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

    def get_ray(self, point):
        """
        :param point: point of type tuple (x: float, y: float), where -1 <= x, y <= 1
        :return:
        """
        fov_x, fov_y = self.fov_angle
        d = 1 / np.tan(fov_x / 2)
        x, y = point
        u, v, w = self.direction
        aspect = fov_y / fov_x

        dir = x * u + aspect * y * v + d * w
        normalized_direction = dir / np.sqrt(np.dot(dir, dir))

        return Ray(self.origin, normalized_direction)

    def to_normalized(self, point):
        """
        :param point: tuple (x: int, y: int) where x in (0, resolution_x - 1) and y in (0, resolution_y - 1)
        :return: point with type tuple (x: float, y: float), where -1 <= x,y <= 1
        """
        x, y = point
        x_norm = 2 * x / self.resolution_x - 1
        y_norm = 2 * y / self.resolution_y - 1

        return np.array([x_norm, y_norm])