"""Simple utility functions."""
import numpy as np
import statsmodels.api as sm

from itertools import chain
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def lin_mod(xs, y):
    """Run a linear model."""
    xs = np.stack(xs, 1)
    xs_ = StandardScaler().fit_transform(xs)
    xs_ = sm.add_constant(xs_)
    model = sm.OLS(y, xs_).fit()
    return model


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


def gen_codes(letters, size, generate):
    """Generate codes which are binomially distributed."""
    codes = generate((len(letters), size))
    return dict(zip(letters, codes.astype(np.float32)))


def gen_position(length, size, generate):
    """Generate random size codes."""
    return generate((length, size))


def generate_both(words, size, generate):
    """Generate both word and position codes at the same time."""
    m = max([len(x) for x in words])
    letters = set(chain.from_iterable(words))
    return gen_codes(letters, size, generate), gen_position(m, size, generate)
