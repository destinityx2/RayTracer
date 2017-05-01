__author__ = 'ivan'

import numpy as np
from primitives import Hit


class Object:
    def intersect(self, ray, tol):
        """
        :param ray: object of type Ray
        :param tol: tolerance (typical value: 10^{-3})
        :return: object of type Hit if there is intersection, None otherwise
        """
        raise RuntimeError("Unsupported operation")


class Sphere(Object):
    def __init__(self, radius, origin):
        self.radius = radius
        self.origin = origin

    def intersect(self, ray, tol):
        origin = ray.origin - self.origin

        # a = 1 in case when ray.direction is unit length
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(origin, ray.direction)
        c = np.dot(origin, origin) - self.radius * self.radius

        delta = b * b - 4 * a * c
        if delta < 0:
            return None

        delta_sqrt = np.sqrt(delta)
        t_1 = (-b - delta_sqrt) / (2 * a)
        t_2 = (-b + delta_sqrt) / (2 * a)

        hit_point = None
        if t_1 > tol:
            hit_point = ray.point(t_1)
        elif t_2 > tol:
            hit_point = ray.point(t_2)
        else:
            return None

        norm = hit_point - self.origin
        normalized_norm = norm / np.sqrt(np.dot(norm, norm))
        return Hit(hit_point, normalized_norm, ray, self)


class Plane(Object):
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal

    def intersect(self, ray, tol):
        D = -np.dot(self.origin, self.normal)

        denominator = np.dot(self.normal, ray.direction)

        if denominator == 0:
            return None

        t = -(D + np.dot(self.normal, ray.origin)) / denominator

        if t < tol:
            return None

        return Hit(ray.point(t), self.normal, ray, self)


class Triangle(Object):
    def __init__(self, p0, p1, p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def intersect(self, ray, tol):
        col1 = self.p0 - self.p1
        col2 = self.p0 - self.p2
        col3 = ray.direction

        A = np.column_stack((col1, col2, col3))
        b = self.p0 - ray.origin

        x = np.dot(np.linalg.inv(A), b)
        beta = x[0]
        gamma = x[1]
        t = x[2]

        if t < tol:
            return None

        if beta > 0 and gamma > 0 and beta + gamma < 1:
            norm = np.cross(self.p1 - self.p0, self.p2 - self.p0)
            norm = norm / np.sqrt(np.dot(norm, norm))
            return Hit(ray.point(t), norm, ray, self)

        return None
