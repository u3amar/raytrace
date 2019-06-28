from PIL import Image
import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        return self.origin + t * self.direction


class HitRecord:
    def __init__(self, t, p, normal):
        self.t = t
        self.p = p
        self.normal = normal


class Hitable:
    def hit(ray, t_min, t_max):
        return False


class Sphere(Hitable):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

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
            return HitRecord(hit_t, hit_loc, s_norm)


class World:
    def __init__(self, hitables):
        self.hitables = hitables

    def add_hitables(self, hitables):
        self.hitables += hitables

    def hit(self, ray, t_min, t_max):
        closest_hit = None
        closest_distance = t_max
        for h in self.hitables:
            rec = h.hit(ray, t_min, closest_distance)
            if rec:
                closest_distance = rec.t
                closest_hit = rec
        return closest_hit


class Camera:
    def __init__(self):
        self.lower_left_corner = np.array([-2.0, -1.0, -1.0])
        self.horizontal = np.array([4.0, 0.0, 0.0])
        self.vertical = np.array([0.0, 2.0, 0.0])
        self.origin = np.array([0.0, 0.0, 0.0])

    def get_ray(self, u, v):
        ray_offset = u * self.horizontal + v * self.vertical
        return Ray(self.origin,
                   self.lower_left_corner + ray_offset - self.origin)


def color(ray, world):
    MAX_DIST = 1000000000
    hit_rec = world.hit(ray, 0.0, MAX_DIST)

    if hit_rec:
        return .5 * (hit_rec.normal + 1)

    norm_vec = ray.direction / np.linalg.norm(ray.direction)
    t = .5 * (norm_vec[1] + 1.0)
    c1 = np.array([1.0, 1.0, 1.0])
    c2 = np.array([.5, .7, 1.0])
    return (1.0 - t) * c1 + t * c2


if __name__ == '__main__':
    scale = 4.0
    base_im_width = 200
    base_im_height = 100
    n_samples_in_pixel = 5

    im_width = int(base_im_width * scale)
    im_height = int(base_im_height * scale)
    im_arr = np.zeros((im_height, im_width, 3), dtype=np.uint8)

    s1 = Sphere(np.array([0.0, 0.0, -1.0]), .5)
    s2 = Sphere(np.array([0.0, -100.5, -1.0]), 100)
    world = World([s1, s2])

    cam = Camera()
    for j in range(im_height):
        for x in range(im_width):
            y = im_height - j
            col = np.array([0.0, 0.0, 0.0])
            for s in range(n_samples_in_pixel):
                u = (x + np.random.rand()) / im_width
                v = (y + np.random.rand()) / im_height

                r = cam.get_ray(u, v)
                col += color(r, world)

            col /= n_samples_in_pixel
            im_arr[j, x] = np.array(255.0 * col, dtype=np.uint8)

    im = Image.fromarray(im_arr, 'RGB')
    im.show()
