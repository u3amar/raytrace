from PIL import Image
from multiprocessing import Pool
import numpy as np


def random_in_unit_sphere():
    # Sample from uniform distribution and push values
    # within the range [-1, 1)
    loc = 2.0 * np.random.rand(3) - 1.0
    while np.linalg.norm(loc) >= 1.0:
        loc = 2.0 * np.random.rand(3) - 1.0
    return loc


def vec3(a, b, c):
    return np.array([a, b, c])


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        return self.origin + t * self.direction


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


class ScatterResult:
    def __init__(self, scattered_ray, attenuation):
        self.scattered_ray = scattered_ray
        self.attenuation = attenuation


class Material:
    def scatter(self, in_ray, attenutation):
        return


class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, in_ray, hit_rec):
        target = hit_rec.normal + random_in_unit_sphere()
        scattered_ray = Ray(hit_rec.p, target)
        return ScatterResult(scattered_ray, self.albedo)


class Metal(Material):
    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        self.fuzz = max(1.0, fuzz)

    def scatter(self, in_ray, hit_rec):
        unit_dir = in_ray.direction / np.linalg.norm(in_ray.direction)
        ref_r = self.reflect(unit_dir, hit_rec.normal)
        ref_r += self.fuzz * random_in_unit_sphere()

        if ref_r.dot(hit_rec.normal) <= 0:
            return

        scattered = Ray(hit_rec.p, ref_r)
        return ScatterResult(scattered, self.albedo)

    def reflect(self, source, normal):
        return source - 2 * source.dot(normal) * normal


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
        self.lower_left_corner = vec3(-2.0, -1.0, -1.0)
        self.horizontal = vec3(4.0, 0.0, 0.0)
        self.vertical = vec3(0.0, 2.0, 0.0)
        self.origin = vec3(0.0, 0.0, 0.0)

    def get_ray(self, u, v):
        ray_offset = u * self.horizontal + v * self.vertical
        return Ray(self.origin,
                   self.lower_left_corner + ray_offset - self.origin)


class RayTraceOperation:
    def __init__(self, x, y, world, camera):
        self.x = x
        self.y = y
        self.world = world
        self.camera = camera

    def color(self, ray, depth=0):
        MAX_DIST = 1000000000
        hit_rec = self.world.hit(ray, 0.00001, MAX_DIST)

        if hit_rec:
            scat_res = hit_rec.material.scatter(ray, hit_rec)
            if depth < 50 and scat_res:
                return scat_res.attenuation * self.color(
                    scat_res.scattered_ray, depth + 1)
            else:
                return vec3(0.0, 0.0, 0.0)

        norm_vec = ray.direction / np.linalg.norm(ray.direction)
        t = .5 * (norm_vec[1] + 1.0)
        c1 = vec3(1.0, 1.0, 1.0)
        c2 = vec3(.5, .7, 1.0)
        return (1.0 - t) * c1 + t * c2

    def compute_pixel_color(self):
        col = vec3(0.0, 0.0, 0.0)
        for s in range(n_samples_in_pixel):
            u = (self.x + np.random.rand()) / im_width
            v = (self.y + np.random.rand()) / im_height

            r = self.camera.get_ray(u, v)
            col += self.color(r)

        col = 255.0 * np.sqrt(col / n_samples_in_pixel)
        return np.array(col, dtype=np.uint8)


def compute_pixel_color(ray_trace_op):
    return ray_trace_op.compute_pixel_color()


if __name__ == '__main__':
    scale = 4.0
    base_im_width = 200
    base_im_height = 100
    n_samples_in_pixel = 50
    num_processors = 16

    im_width = int(base_im_width * scale)
    im_height = int(base_im_height * scale)
    im_arr = np.zeros((im_height, im_width, 3), dtype=np.uint8)

    s1 = Sphere(vec3(0.0, 0.0, -1.0),
                .5,
                Lambertian(vec3(.8, .3, .3)))

    s2 = Sphere(vec3(0.0, -100.5, -1.0),
                100,
                Lambertian(vec3(.8, .8, 0.0)))

    s3 = Sphere(vec3(1.0, 0.0, -1.0),
                .5,
                Metal(vec3(.8, .6, .2), .3))

    s4 = Sphere(vec3(-1.0, 0.0, -1.0),
                .5,
                Metal(vec3(.8, .8, .8), 1.0))

    world = World([s1, s2, s3, s4])

    cam = Camera()
    ray_trace_args = []
    for j in range(im_height):
        for x in range(im_width):
            y = im_height - j
            op = RayTraceOperation(x, y, world, cam)
            ray_trace_args.append(op)

    with Pool(num_processors) as p:
        cols_per_pixel = p.map(compute_pixel_color, ray_trace_args)
        for op, color in zip(ray_trace_args, cols_per_pixel):
            j = im_height - op.y
            im_arr[j, op.x] = color

    im = Image.fromarray(im_arr, 'RGB')
    im.show()
