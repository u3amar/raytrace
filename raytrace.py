import numpy as np
from PIL import Image
from multiprocessing import Pool
from utils import vec3
from hitable import HitRecord, Hitable, Sphere
from camera import Camera
from material import Lambertian, Metal
from ray import Ray
from typing import Optional, List


class World:
    def __init__(self, hitables: List[Hitable]):
        self.hitables = hitables

    def add_hitables(self, hitables: List[Hitable]):
        self.hitables += hitables

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest_hit = None
        closest_distance = t_max
        for h in self.hitables:
            rec = h.hit(ray, t_min, closest_distance)
            if rec:
                closest_distance = rec.t
                closest_hit = rec
        return closest_hit


class RayTraceOperation:
    def __init__(self,
                 x: int,
                 y: int,
                 n_samples_in_pixel: int,
                 world: World,
                 camera: Camera):
        self.x = x
        self.y = y
        self.n_samples_in_pixel = n_samples_in_pixel
        self.world = world
        self.camera = camera

    def color(self, ray: Ray, depth: int = 0):
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
        for s in range(self.n_samples_in_pixel):
            u = (self.x + np.random.rand()) / im_width
            v = (self.y + np.random.rand()) / im_height

            r = self.camera.get_ray(u, v)
            col += self.color(r)

        col = 255.0 * np.sqrt(col / n_samples_in_pixel)
        return np.array(col, dtype=np.uint8)


def compute_pixel_color(ray_trace_op: RayTraceOperation):
    return ray_trace_op.compute_pixel_color()


if __name__ == '__main__':
    scale = 1.0
    base_im_width = 200
    base_im_height = 100
    n_samples_in_pixel = 5
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
            op = RayTraceOperation(x, y, n_samples_in_pixel, world, cam)
            ray_trace_args.append(op)

    with Pool(num_processors) as p:
        cols_per_pixel = p.map(compute_pixel_color, ray_trace_args)
        for op, color in zip(ray_trace_args, cols_per_pixel):
            j = im_height - op.y
            im_arr[j, op.x] = color

    im = Image.fromarray(im_arr, 'RGB')
    im.show()
