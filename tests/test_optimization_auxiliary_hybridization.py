# %%
import pytest
import numpy as np
import src.frequency_greens_function as fg
from src import optimization_auxiliary_hybridization as opt_aux
import src.auxiliary_system_parameter as auxp
from src import dos_util as du

# %%


def test_const_function_values():
    freq = np.linspace(-5, 5, 1001)
    targed_hyb = fg.FrequencyGreen(
        freq, 1j * np.ones(freq.shape), 1j * np.ones(freq.shape))
    aux_hyb = fg.FrequencyGreen(freq)

    renormalized_to_one = opt_aux.cost_function(targed_hyb, aux_hyb) == 1.0
    weight = 2 * np.ones(freq.shape)

    weight_times2 = opt_aux.cost_function(
        targed_hyb, aux_hyb, weight=weight) == 2.0
    weight = np.array([2.0 * du.heaviside(w, 0) for w in freq])

    aux_hyb = fg.FrequencyGreen(
        freq, 1j * np.ones(freq.shape), 1j * np.ones(freq.shape))

    is_equal = opt_aux.cost_function(targed_hyb, aux_hyb) == 0.0
    assert renormalized_to_one and weight_times2 and (
        is_equal)


def test_optimization_ph_symmertry():
    beta = 100
    N_freq = 1000
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
    options = {"disp": True, "maxiter": 500, 'ftol': 1e-5}
    try:
        Nb = 1
        x_start = [3., 0.81, 0.5, -1.40, 0.2]
        result_nb1 = opt_aux.optimization_ph_symmertry(Nb, hybridization,
                                                       x_start=x_start,
                                                       options=options)
        hyb_aux_nb1 = opt_aux.get_aux_hyb(result_nb1.x, Nb, freq)
    except ValueError:
        print(f"Minimization for Nb = {Nb}, not converged.")
    hyb_start_nb1 = opt_aux.get_aux_hyb(x_start, Nb, freq)
    assert (opt_aux.cost_function(hybridization, hyb_aux_nb1)
            < opt_aux.cost_function(hybridization, hyb_start_nb1))


def test_get_aux_hyb():
    Nb = 1
    freq = np.linspace(-5, 5, 1001)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.1 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])

    aux = auxp.AuxiliarySystem(Nb, freq)
    aux.set_ph_symmetric_aux(es, ts, gamma)

    green = fg.FrequencyGreen(aux.ws)
    green.set_green_from_auxiliary(aux)
    hyb_explicit = green.get_self_enerqy()
    hyb = opt_aux.get_aux_hyb([*es, *ts, *gamma], Nb, freq)

    diff_hyb_ret = np.abs(hyb_explicit.retarded -
                          hyb.retarded)
    diff_hyb_ret = diff_hyb_ret[~np.isnan(diff_hyb_ret)]

    diff_hyb_kel = np.abs(hyb_explicit.keldysh -
                          hyb.keldysh)
    diff_hyb_kel = diff_hyb_kel[~np.isnan(diff_hyb_kel)]

    assert (np.allclose(diff_hyb_kel, np.zeros(diff_hyb_kel.shape)) and
            np.allclose(diff_hyb_ret, np.zeros(diff_hyb_ret.shape)))


if __name__ == "__main__":

    pytest.main("-v test_frequency_greens_function.py")

# %%
