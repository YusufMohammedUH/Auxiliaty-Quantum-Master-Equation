# %%
import pytest
import numpy as np
from src import frequency_greens_function as fg

# %%


def test_FrequencySystem_freq_raise_TypeError():
    ws = [i for i in range(10)]
    with pytest.raises(TypeError):
        fg.FrequencyGreen(ws)


def test_FrequencySystem_retarded_raise_ValueError():
    ws = [i for i in range(10)]
    with pytest.raises(ValueError):
        fg.FrequencyGreen(np.array(ws), np.array(ws[2:]))


def test_FrequencySystem_keldysh_raise_ValueError():
    ws = [i for i in range(10)]
    with pytest.raises(ValueError):
        fg.FrequencyGreen(np.array(ws), np.array(ws), np.array(ws[2:]))

# %%


def flat_dos(w, D, gamma):
    x = D + w
    y = (w - D)
    if x == 0:
        real_part = float("inf")
    elif y == 0:
        real_part = float("-inf")
    else:
        real_part = (gamma / np.pi) * np.log(np.abs(y / x))

    imag_part = 0
    if np.abs(w) < D:
        imag_part = gamma
    return complex(real_part, imag_part)


def heaviside(x, x0):
    if x < x0:
        return 0.
    elif x == x0:
        return 0.5
    else:
        return 1.


def test_get_self_enerqy_and_dyson():
    ws = np.linspace(-4, 4, 1001)
    flat_hybridization_retarded = np.array([flat_dos(w, 3.0, 1.0) for w in ws])
    flat_hybridization_keldysh = np.array(
        [1.j * (1. / np.pi) * (1. - 2. * heaviside(w, 0)) *
            np.imag(flat_dos(w, 3.0, 1.0)) for w in ws])
    hybridization = fg.FrequencyGreen(
        ws, flat_hybridization_retarded, flat_hybridization_keldysh)
    green = fg.FrequencyGreen(ws)
    green.dyson(green.freq, hybridization)
    hybridization_retreaved = green.get_self_enerqy()
    assert (np.allclose(hybridization_retreaved.retarded.imag,
                        hybridization.retarded.imag) and
            np.allclose(hybridization_retreaved.retarded.real,
            hybridization.retarded.real) and
            np.allclose(hybridization_retreaved.keldysh,
            hybridization.keldysh))

# TODO: test for method set_green_from_auxiliary in FrequencyGreen


# %%
if __name__ == "__main__":

    pytest.main("-v test_frequency_greens_function.py")
