import pytest
import numpy as np
from src.greens_function import frequency_greens_function as fg
from src.greens_function import dos_util as du


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


def test_get_self_enerqy_and_dyson():
    beta = 100
    N_freq = 1001
    freq_max = 4
    D = 3
    gamma = 1
    e0 = 0
    mu = 0
    freq = np.linspace(-freq_max, freq_max, N_freq)
    flat_hybridization_retarded = np.array(
        [du.flat_bath_retarded(w, 0, D, gamma) for w in freq])
    flat_hybridization_keldysh = np.array(
        [1.j * (1. / np.pi) * (1. - 2. * du.fermi(w, e0, mu, beta)) *
            np.imag(du.flat_bath_retarded(w, 0, D, gamma)) for w in freq])
    hybridization = fg.FrequencyGreen(
        freq, flat_hybridization_retarded, flat_hybridization_keldysh)
    green = fg.FrequencyGreen(freq)
    green.dyson(self_energy=hybridization)
    hybridization_retreaved = green.get_self_enerqy()

    diff_hyb_ret = np.abs(hybridization_retreaved.retarded -
                          hybridization.retarded)
    diff_hyb_ret = diff_hyb_ret[~np.isnan(diff_hyb_ret)]

    diff_hyb_kel = np.abs(hybridization_retreaved.keldysh -
                          hybridization.keldysh)
    diff_hyb_kel = diff_hyb_kel[~np.isnan(diff_hyb_kel)]

    assert (np.allclose(diff_hyb_kel, np.zeros(diff_hyb_kel.shape)) and
            np.allclose(diff_hyb_ret, np.zeros(diff_hyb_ret.shape)))

# TODO: test for method set_green_from_auxiliary in FrequencyGreen


if __name__ == "__main__":

    pytest.main("-v test_frequency_greens_function.py")
