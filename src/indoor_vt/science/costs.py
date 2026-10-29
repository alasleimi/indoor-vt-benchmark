"""Injected numerical functions, named by their operation."""

import numpy as np


def absolute_difference(a, b):
    return np.abs(a - b)


def squared_difference(a, b):
    return np.square(a - b)


def mean(values):
    return np.mean(values, axis=0)


def median(values):
    return np.median(values, axis=0)


def squared_loss(z):
    return np.vstack((z, np.ones_like(z), np.zeros_like(z)))


def soft_l1_loss(z):
    root = np.sqrt(1 + z)
    return np.vstack((2 * (root - 1), 1 / root, -0.5 / root**3))


def gaussian_weights(errors, scale_m):
    return np.exp(-0.5 * (errors / scale_m)**2)
