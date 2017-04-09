__author__ = 'ivan'


class Scene:
    def __init__(self, objects, lights):
        """
        :param objects: list of objects of type Object
        :param lights: list of lights of type Light
        :return: None
        """
        self.objects = objects
        self.lights = lights

    def n_objects(self):
        return len(self.objects)

    def n_lights(self):
        return len(self.lights)

    def get_light(self, i):
        return self.lights[i]

    def get_object(self, i):
        return self.objects[i]
