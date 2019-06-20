"""Splatter (kanerva)."""
import numpy as np
from .utils import compose, generate_both, memoize
from functools import partial


def generate(size):
    return np.random.uniform(size=size) > .5


def majority(vecs):
    """Majority."""
    s = np.mean(vecs, 0)
    b = np.flatnonzero(s == .5)
    s[b] = np.random.rand(len(b))
    return s > .5


def encode(x, y):
    r = np.logical_xor(x, y)
    if np.ndim(r) == 1:
        r = r[None, :]
    return r


def decode(x, y):
    return encode(x, y)


compose = partial(compose, adder=majority, encoder=encode)
generate_both = partial(generate_both, generate=generate)
