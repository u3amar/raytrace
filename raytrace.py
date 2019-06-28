from PIL import Image
import numpy as np


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        return self.origin + t * self.direction


def hit_sphere(center, radius, ray):
    r = ray.origin - center

    a = ray.direction.dot(ray.direction)
    b = 2.0 * ray.direction.dot(r)
    c = r.dot(r) - radius ** 2

    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return -1.0

    return (-b - np.sqrt(disc)) / (2.0 * a)


def color(ray):
    sphere_center = np.array([0.0, 0.0, -1.0])
    sphere_intersect = hit_sphere(sphere_center, .5, r)
    if sphere_intersect > 0.0:
        sphere_ray = ray.point_at_parameter(sphere_intersect)
        s_norm = sphere_ray - sphere_center
        s_norm /= np.linalg.norm(s_norm)
        return .5 * (s_norm + 1)
            
    norm_vec = ray.direction / np.linalg.norm(ray.direction)
    t = .5 * (norm_vec[1] + 1.0)
    c1 = np.array([1.0, 1.0, 1.0])
    c2 = np.array([.5, .7, 1.0])
    return (1.0 - t) * c1 + t * c2


if __name__ == '__main__':
    im_width = 400
    im_height = 200
    im_arr = np.zeros((im_height, im_width, 3), dtype=np.uint8)

    lower_left_corner = np.array([-2.0, -1.0, -1.0])
    horizontal = np.array([4.0, 0.0, 0.0])
    vertical = np.array([0.0, 2.0, 0.0])
    origin = np.array([0.0, 0.0, 0.0])

    for j in range(im_height):
        for x in range(im_width):
            y = im_height - j

            u = x / im_width
            v = y / im_height
            ray_offset = u * horizontal + v * vertical
            r = Ray(origin, lower_left_corner + ray_offset)

            im_arr[j, x] = np.array(255.0 * color(r), dtype=np.uint8)

    im = Image.fromarray(im_arr, 'RGB')
    im.show()
