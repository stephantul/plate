"""Splatter (kanerva)."""
import numpy as np
from .utils import compose, generate_both
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
    return np.logical_xor(x, y)


def decode(x, y):
    return encode(x, y)


compose = partial(compose, adder=majority, encoder=encode)
generate_both = partial(generate_both, generate=generate)
