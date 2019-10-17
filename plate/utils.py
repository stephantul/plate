"""Simple utility functions."""
import numpy as np

from itertools import chain
from tqdm import tqdm


def memoize(function):
    """
    Simple memoization.

    This assumes all arguments implement a.tostring() function.
    This holds for numpy arrays.
    """
    memo = {}

    def new_function(*args, **kwargs):
        try:
            return memo[tuple([x.tostring() for x in args])]
        except KeyError:
            r = function(*args, **kwargs)
            memo[tuple([x.tostring() for x in args])] = r
            return r
    return new_function


def compose(words, letter_codes, position_codes, adder, encoder):
    """Generate word representations through letter and position codes."""
    for x in tqdm(words):
        code = []
        for idx, letter in enumerate(x):
            r = encoder(letter_codes[letter],
                        position_codes[idx])[0]
            code.append(r)

        yield adder(code)


def compose_bigrams(words, letter_codes, position_codes, adder, encoder):
    """Generate letter and position codes."""
    for x in tqdm(words):
        code = []
        for idx, letter in enumerate(x):
            r = encoder(letter_codes[letter],
                        position_codes[idx])[0]
            code.append(r)

        yield adder(code)


def gen_codes(words, size, generate):
    """Generate codes."""
    letters = set(chain.from_iterable(words))
    letters.add(" ")
    codes = generate((len(letters), size))
    return dict(zip(letters, codes.astype(np.float32)))


def gen_position(length, size, generate):
    """Generate random size codes."""
    return generate((length, size))


def generate_both(words, size, generate):
    """Generate both word and position codes at the same time."""
    m = max([len(x) for x in words])
    return gen_codes(words, size, generate), gen_position(m, size, generate)
