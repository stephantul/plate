"""Circular convolution and involution."""
import numpy as np
from .utils import compose, generate_both
from functools import partial


def generate(size):
    return np.random.normal(size=size)


def addition(p):
    return np.sum(p, 0)


def circular_convolution(x, y):
    """Make a fast version."""
    if np.ndim(x) == 1:
        x = x[None, :]
    if np.ndim(y) == 1:
        y = y[None, :]
    assert(x.shape == y.shape)
    length = x.shape[1]
    vec = np.arange(length)
    mtr = (vec[None, :] - vec[:, None]) % length
    return (x[:, :, None] * y[:, mtr]).sum(-2)


def involution(x):
    """Involution operator."""
    if np.ndim(x) == 1:
        x = x[None, :]
    return np.concatenate([x[:, None, 0], x[:, -1:0:-1]], 1)


def decode(a, b):
    """Decode a vector using involution."""
    return circular_convolution(involution(a), b)


compose = partial(compose, adder=addition, encoder=circular_convolution)
generate_both = partial(generate_both, generate=generate)
