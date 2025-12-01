import numpy as np


def _meshify(t: np.ndarray):
    dim = len(t)
    t_mesh = np.repeat(t, dim).reshape((dim, dim))
    return (t_mesh, t_mesh.T)


def min_kernel(t: np.ndarray):
    s, t = _meshify(t)
    out = np.minimum(s, t)
    return out


def bridge_kernel(t: np.ndarray):
    s, t = _meshify(t)
    out = np.minimum(s, t) - s * t
    return out


def gaussian_kernel(t: np.ndarray, theta=1):
    s, t = _meshify(t)
    out = np.exp(-((s - t) ** 2) / theta)
    return out
