import numpy as np
from scipy.special import logsumexp


def _meshify(z: np.ndarray):
    dim = len(z)
    z_mesh = np.repeat(z, dim).reshape((dim, dim))
    return (z_mesh, z_mesh.T)


def min_kernel(t: np.ndarray):
    s, t = _meshify(t)
    out = np.minimum(s, t)
    return out


def bridge_kernel(t: np.ndarray):
    s, t = _meshify(t)
    out = np.minimum(s, t) - s * t
    return out


def smooth_bridge_kernel(t: np.ndarray, epsilon=1.0):
    s, t = _meshify(t)
    out = -epsilon * logsumexp(np.dstack([-s / epsilon, -t / epsilon]), axis=2) - s * t
    return out


def gaussian_kernel(t: np.ndarray, theta=1):
    s, t = _meshify(t)
    out = np.exp(-((s - t) ** 2) / theta)
    return out


def laplacian_kernel(t: np.ndarray, theta=1.0):
    s, t = _meshify(t)
    out = np.exp(-np.abs(s - t) / theta)
    return out


def matern32_kernel(t: np.ndarray, theta=1.0):
    s, t = _meshify(t)
    d = np.abs(s - t)
    sqrt3_d = np.sqrt(3) * d / theta
    out = (1 + sqrt3_d) * np.exp(-sqrt3_d)
    return out


def polynomial_kernel(t: np.ndarray, degree=2, c=0):
    s, t = _meshify(t)
    out = (s * t + c) ** degree
    return out


def periodic_kernel(t: np.ndarray, period=1.0, theta=1.0):
    s, t = _meshify(t)
    sine_part = np.sin(np.pi * np.abs(s - t) / period) ** 2
    out = np.exp(-2 * sine_part / theta**2)
    return out


def integrated_brownian_kernel(x: np.ndarray):
    s, t = _meshify(t)
    out = s**2 * (t / 2 - s / 6)
    return out


def chi_square_kernel(t: np.ndarray, gamma=1.0):
    """
    Computes the Exponential Chi-Square kernel.
    NOTE: Inputs 't' must be non-negative (t >= 0).
    """
    s, t = _meshify(t)

    numerator = (s - t) ** 2
    denominator = s + t

    # Safe division to handle the case where s=0 and t=0 (0/0 -> 0)
    # If s+t is 0, the distance is 0.
    distance = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    out = np.exp(-gamma * distance)
    return out
