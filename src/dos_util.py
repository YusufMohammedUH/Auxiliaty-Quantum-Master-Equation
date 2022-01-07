import numpy as np
import src.frequency_greens_function as fg


def flat_bath_retarded(w, e0, D, gamma):
    x = D + (w - e0)
    y = (w - e0 - D)
    if x == 0:
        real_part = -float("inf")
    elif y == 0:
        real_part = -float("-inf")
    else:
        real_part = (-gamma / np.pi) * np.log(np.abs(y / x))

    imag_part = 0
    if np.abs(w - e0) < D:
        imag_part = -gamma
    return complex(real_part, imag_part)


def lorenzian_bath_retarded(w, e0, gamma, v=1):
    x = w - e0 - 1j * gamma
    y = (w - e0)**2 + gamma**2
    result = 0
    if y == 0:
        result = complex(0, float("-inf"))
    else:
        result = x / y
    return (v**2) * result


def heaviside(x, x0):
    if x < x0:
        return 0.
    elif x == x0:
        return 0.5
    else:
        return 1.


def fermi(e, e0, mu, beta):
    x = (e + e0 - mu) * beta
    if x < 0.0:
        return 1.0 / (1.0 + np.exp(x))
    else:
        return 1.0 - 1.0 / (1.0 + np.exp(-x))


def set_hybridization(freq, retarded_function, *args):
    e0 = args[0]
    mu = args[1]
    beta = args[2]
    retarded = np.array([retarded_function(f, e0, *args[3:]) for f in freq])
    keldysh = np.array([1j * (1 - 2 * fermi(f, e0, mu, beta)) *
                        np.imag(retarded_function(f, e0, *args[3:]))
                        for f in freq])
    return fg.FrequencyGreen(freq=freq, retarded=retarded, keldysh=keldysh)
