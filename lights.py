__author__ = 'ivan'


class Light:
    def get_intensity(self, ray):
        """
        :param ray: incoming ray to the light source of type Ray
        :return: intensity in the ray direction
        """
        raise RuntimeError("Unsupported operation")


class PointLight(Light):
    def __init__(self, origin, intensity):
        self.origin = origin
        self.intensity = intensity

    def get_intensity(self, ray):
        return self.intensity
