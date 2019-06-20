"""Circular convolution and involution."""
import numpy as np
from .utils import compose, generate_both, memoize
from functools import partial


def generate(size):
    """Generates normally distributed random vectors of size."""
    return np.random.normal(size=size)


def addition(p):
    return np.sum(p, 0)


@memoize
def circular_convolution(x, y):
    """A fast version of the circular convolution."""
    # Stolen from:
    # http://www.indiana.edu/~clcl/holoword/Site/__files/holoword.py
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real


def involution(x):
    """Involution operator."""
    if np.ndim(x) == 1:
        x = x[None, :]
    return np.concatenate([x[:, None, 0], x[:, -1:0:-1]], 1)


def circular_correlation(x, y):
    """Circular correlation is the inverse of circular convolution."""
    return circular_convolution(involution(x), y)


def decode(x, y):
    """Simple renaming."""
    return circular_correlation(x, y)


compose = partial(compose, adder=addition, encoder=circular_convolution)
generate_both = partial(generate_both, generate=generate)
