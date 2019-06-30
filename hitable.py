import numpy as np


class HitRecord:
    def __init__(self, t, p, normal, material):
        self.t = t
        self.p = p
        self.normal = normal
        self.material = material


class Hitable:
    def hit(ray, t_min, t_max):
        return False


class Sphere(Hitable):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        r = ray.origin - self.center

        a = ray.direction.dot(ray.direction)
        b = 2.0 * ray.direction.dot(r)
        c = r.dot(r) - self.radius ** 2

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return

        r1 = (-b - np.sqrt(disc)) / (2.0 * a)
        r2 = (-b + np.sqrt(disc)) / (2.0 * a)

        hit_t = None
        if r1 < t_max and r1 > t_min:
            hit_t = r1
        elif r2 < t_max and r2 > t_max:
            hit_t = r2

        if hit_t:
            hit_loc = ray.point_at_parameter(hit_t)
            s_norm = hit_loc - self.center
            s_norm /= np.linalg.norm(s_norm)
            return HitRecord(hit_t, hit_loc, s_norm, self.material)
