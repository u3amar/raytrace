from utils import vec3
from ray import Ray


class Camera:
    def __init__(self):
        self.lower_left_corner = vec3(-2.0, -1.0, -1.0)
        self.horizontal = vec3(4.0, 0.0, 0.0)
        self.vertical = vec3(0.0, 2.0, 0.0)
        self.origin = vec3(0.0, 0.0, 0.0)

    def get_ray(self, u: float, v: float) -> Ray:
        ray_offset = u * self.horizontal + v * self.vertical
        return Ray(self.origin,
                   self.lower_left_corner + ray_offset - self.origin)
