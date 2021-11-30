import numpy as np


def flat_dos(w, D, gamma):
    x = D + w
    y = (w - D)
    if x == 0:
        real_part = -float("inf")
    elif y == 0:
        real_part = -float("-inf")
    else:
        real_part = (-gamma / np.pi) * np.log(np.abs(y / x))

    imag_part = 0
    if np.abs(w) < D:
        imag_part = -gamma
    return complex(real_part, imag_part)


def heaviside(x, x0):
    if x < x0:
        return 0.
    elif x == x0:
        return 0.5
    else:
        return 1.


def fermi(e, beta):
    x = e * beta
    if x < 0.0:
        return 1.0 / (1.0 + np.exp(x))
    else:
        return 1.0 - 1.0 / (1.0 + np.exp(-x))
