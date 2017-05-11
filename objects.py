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

    def all_intersections(self, ray):
        origin = ray.origin - self.origin

        # a = 1 in case when ray.direction is unit length
        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(origin, ray.direction)
        c = np.dot(origin, origin) - self.radius * self.radius

        delta = b * b - 4 * a * c
        if delta < 0:
            return []

        delta_sqrt = np.sqrt(delta)
        t_1 = (-b - delta_sqrt) / (2 * a)
        t_2 = (-b + delta_sqrt) / (2 * a)

        return [t_1, t_2]

    def get_hit(self, ray, t):
        hit_point = ray.point(t)
        norm = hit_point - self.origin
        normalized_norm = norm / np.sqrt(np.dot(norm, norm))
        if np.dot(normalized_norm, ray.direction) > 0:
            normalized_norm *= -1
        return Hit(hit_point, normalized_norm, ray, self)

    def intersect(self, ray, tol):
        ts = self.all_intersections(ray)
        for t in ts:
            if t > tol:
                return self.get_hit(ray, t)

        return None


class Plane(Object):
    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal

    def all_intersections(self, ray):
        D = -np.dot(self.origin, self.normal)

        denominator = np.dot(self.normal, ray.direction)

        if denominator == 0:
            return []

        t = -(D + np.dot(self.normal, ray.origin)) / denominator

        return [t]

    def intersect(self, ray, tol):
        ts = self.all_intersections(ray)
        for t in ts:
            if t > tol:
                normal = self.normal
                if np.dot(normal, ray.direction) > 0:
                    normal *= -1
                return Hit(ray.point(t), normal, ray, self)

        return None


class Triangle(Object):
    def __init__(self, p0, p1, p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    def all_intersections(self, ray):
        col1 = self.p0 - self.p1
        col2 = self.p0 - self.p2
        col3 = ray.direction

        A = np.column_stack((col1, col2, col3))
        b = self.p0 - ray.origin

        x = np.dot(np.linalg.pinv(A), b)
        beta = x[0]
        gamma = x[1]
        t = x[2]

        if beta > 0 and gamma > 0 and beta + gamma < 1:
            return [t]

        return []

    def intersect(self, ray, tol):
        ts = self.all_intersections(ray)
        for t in ts:
            if t > tol:
                norm = np.cross(self.p1 - self.p0, self.p2 - self.p0)
                norm = norm / np.sqrt(np.dot(norm, norm))
                if np.dot(norm, ray.direction) > 0:
                    norm *= -1

                return Hit(ray.point(t), norm, ray, self)

        return None


class CSGTree:
    def __init__(self, root):
        self.root = root


class CSGNode:
    UNION_OPERATION = "union"
    INTERSECTION_OPERATION = "intersection"
    DIFFERENCE_OPERATION = "difference"

    def __init__(self, op=None, obj=None, left=None, right=None):
        self.op = op
        self.obj = obj
        self.left = left
        self.right = right


class CSG(Object):
    def __init__(self, csg_tree):
        self.csg_tree = csg_tree

    def intersect(self, ray, tol):
        intervals = self._find_intervals(self.csg_tree.root, ray)

        min_t = 1e8
        min_obj = None

        for interval in intervals:
            [(t0, obj0), (t1, obj1)] = interval
            if tol < t0 < min_t:
                min_t = t0
                min_obj = obj0

            if tol < t1 < min_t:
                min_t = t1
                min_obj = obj1

        if min_obj is None:
            return None

        return min_obj.get_hit(ray, min_t)

    def _find_intervals(self, node, ray):
        res = []
        if node is None:
            return res

        left_intervals = self._find_intervals(node.left, ray)
        right_intervals = self._find_intervals(node.right, ray)

        if node.op is not None:
            op = node.op
            if op == CSGNode.UNION_OPERATION:
                res.extend(self._union_op(left_intervals, right_intervals))
            elif op == CSGNode.INTERSECTION_OPERATION:
                res.extend(self._intersection_op(left_intervals, right_intervals))
            elif op == CSGNode.DIFFERENCE_OPERATION:
                res.extend(self._difference_op(left_intervals, right_intervals))
            else:
                raise RuntimeError("{} is unsupported type of operation for CSG".format(op))

        if node.obj is not None:
            obj = node.obj
            ts = obj.all_intersections(ray)
            if len(ts) > 0 and len(ts) != 2:
                raise RuntimeError("CSG: there are less than 2 intersections")

            if len(ts) != 0:
                res.append([(ts[0], obj), (ts[1], obj)])

        return res

    def _union_op(self, left_intervals, right_intervals):
        ll = left_intervals.copy()
        ll.extend(right_intervals)
        return ll

    def _intersection_op(self, left_intervals, right_intervals):
        if len(right_intervals) == 0:
            return []
        res = []
        for left_interval in left_intervals:
            tmp = left_interval
            for right_interval in right_intervals:
                if len(tmp) == 0:
                    break

                [(t0, obj0), (t1, obj1)] = tmp
                [(t2, obj2), (t3, obj3)] = right_interval
                m0 = max(t0, t2)
                m1 = min(t1, t3)
                if m0 > m1:
                    # There is no intersection
                    tmp = []
                else:
                    new_obj0 = obj0 if t0 > t2 else obj2
                    new_obj1 = obj1 if t1 < t3 else obj3
                    tmp = [(m0, new_obj0), (m1, new_obj1)]

            if len(tmp) > 0:
                res.append(tmp)

        return res

    def _difference_op(self, left_intervals, right_intervals):
        if len(right_intervals) == 0:
            return left_intervals
        res = []
        for left_interval in left_intervals:
            tmp = left_interval
            for right_interval in right_intervals:
                if len(tmp) == 0:
                    break

                [(t0, obj0), (t1, obj1)] = tmp
                [(t2, obj2), (t3, obj3)] = right_interval
                m0 = max(t0, t2)
                m1 = min(t1, t3)
                if m0 > m1:
                    # There is no intersection
                    pass
                else:
                    new_t0, new_t1 = 0, 0
                    if t0 < m0 < m1 < t1:
                        # intersection inside
                        tmp = []
                        left_intervals.append([(t0, obj0), (m0, obj1)])
                        left_intervals.append([(m1, obj0), (t1, obj1)])
                        break
                    elif _eq(t0, m0) and _eq(m1, t1) and t0 < t1:
                        tmp = []
                        break
                    elif _eq(t0, m0) and t0 < m1 < t1:
                        new_t0 = m1
                        new_t1 = t1
                    elif t0 < m0 < m1 and _eq(m1, t1):
                        new_t0 = t0
                        new_t1 = m0
                    elif _eq(m0, m1):
                        new_t0 = t0
                        new_t1 = t1
                    else:
                        print("Not supported case: ({},{},{},{}), ({},{})".format(t0, m0, m1, t1, _eq(t0, m0), _eq(m1, t1)))

                    tmp = [(new_t0, obj0), (new_t1, obj1)]

            if len(tmp) > 0:
                res.append(tmp)

        return res


def _eq(x, y, tol=1e-3):
    return np.abs(x - y) < tol
