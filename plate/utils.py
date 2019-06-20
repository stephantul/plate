"""Simple utility functions."""
import numpy as np
from itertools import chain
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def lin_mod(xs, y):
    """Run a linear model."""
    xs = np.stack(xs, 1)
    xs_ = StandardScaler().fit_transform(xs)
    xs_ = sm.add_constant(xs_)
    model = sm.OLS(y, xs_).fit()
    return model


def compose(words, letter_codes, position_codes, adder, encoder):
    """Generate letter and position codes."""
    # calculate codes once
    memo = {}
    for x in tqdm(words):
        code = []
        for idx, letter in enumerate(x):
            try:
                r = memo[(idx, letter)]
            except KeyError:
                r = encoder(position_codes[idx], letter_codes[letter])[0]
                memo[(idx, letter)] = r
            code.append(r)
        yield adder(code)


def gen_codes(letters, size, generate):
    """Generate codes which are binomially distributed."""
    codes = generate((len(letters), size))
    return dict(zip(letters, codes.astype(np.int32)))


def gen_position(length, size, generate):
    """Generate random size codes."""
    return generate((length, size))


def generate_both(words, size, generate):
    """Generate both word and position codes at the same time."""
    m = max([len(x) for x in words])
    letters = set(chain.from_iterable(words))
    return gen_codes(letters, size, generate), gen_position(m, size, generate)
