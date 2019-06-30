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


def unit(vec):
    return vec / np.linalg.norm(vec)
