"""Splatter (kanerva)."""
import numpy as np
from .utils import compose, generate_both, memoize
from functools import partial


def generate(size):
    """Generate binary vectors by generating uniform samples."""
    return np.random.uniform(size=size) >= .5


def majority(vecs):
    """Weird binary vector addition."""
    s = np.mean(vecs, 0)
    b = np.flatnonzero(s == .5)
    s[b] = np.random.rand(len(b))
    return s > .5


@memoize
def encode(x, y):
    """Encoding is done by XOR function."""
    r = np.logical_xor(x, y)
    if np.ndim(r) == 1:
        r = r[None, :]
    return r


def decode(x, y):
    """XOR is an involution, so the inverse of XOR is also XOR."""
    return encode(x, y)


compose = partial(compose, adder=majority, encoder=encode)
generate_both = partial(generate_both, generate=generate)
