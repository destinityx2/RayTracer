__author__ = 'ivan'

import numpy as np


class Ray:
    def __init__(self, origin, direction, to_normalize=True):
        """
        :param origin: tuple (x: float, y: float, z: float)
        :param direction: tuple (x: float, y: float, z: float)
        :return: None
        """
        self.origin = origin
        self.direction = direction
        if to_normalize:
            self.direction /= np.sqrt(np.dot(self.direction, self.direction))

    def point(self, t):
        return self.origin + t * self.direction


class Hit:
    def __init__(self, point, normal, ray, object):
        self.point = point
        self.normal = normal
        self.ray = ray
        self.object = object

    def distance(self):
        d = self.point - self.ray.origin
        return np.sqrt(np.dot(d, d))
