import numpy as np
from ray import Ray
from utils import random_in_unit_sphere
from hitable import HitRecord
from typing import Optional


class ScatterResult:
    def __init__(self, scattered_ray: Ray, attenuation):
        self.scattered_ray = scattered_ray
        self.attenuation = attenuation


class Material:
    def scatter(self,
                in_ray: Ray,
                hit_rec: HitRecord) -> Optional[ScatterResult]:
        return None


class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self,
                in_ray: Ray,
                hit_rec: HitRecord) -> Optional[ScatterResult]:
        target = hit_rec.normal + random_in_unit_sphere()
        scattered_ray = Ray(hit_rec.p, target)
        return ScatterResult(scattered_ray, self.albedo)


class Metal(Material):
    def __init__(self, albedo, fuzz: float):
        self.albedo = albedo
        self.fuzz = max(1.0, fuzz)

    def scatter(self,
                in_ray: Ray,
                hit_rec: HitRecord) -> Optional[ScatterResult]:
        unit_dir = in_ray.direction / np.linalg.norm(in_ray.direction)
        ref_r = self.reflect(unit_dir, hit_rec.normal)
        ref_r += self.fuzz * random_in_unit_sphere()

        if ref_r.dot(hit_rec.normal) <= 0:
            return None

        scattered = Ray(hit_rec.p, ref_r)
        return ScatterResult(scattered, self.albedo)

    def reflect(self, source, normal):
        return source - 2 * source.dot(normal) * normal
