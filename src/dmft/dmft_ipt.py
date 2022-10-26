# %%
from typing import Dict, Union
import numpy as np
import src.dmft.dmft as dmft
import src.greens_function.frequency_greens_function as fg
import src.greens_function.dos_util as du
import src.greens_function.convert_keldysh_components as conv
import src.util.fourier as dft
import matplotlib.pyplot as plt


class DMFT_IPT(dmft.DMFTBase):
    """Implementation of DMFT with the iterated perturbation theory (IPT)
    solver.

    Parameters
    ----------
    dmft.DMFTBase : dmft.DMFTBase
        Parent class in which the solver implemented here is used to solve
        DMFT calculations.
    """

    def __init__(self, parameters: Dict,
                 hyb_leads: Union[fg.FrequencyGreen, None] = None,
                 keldysh_comp: str = "lesser") -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        super().__init__(parameters, hyb_leads, keldysh_comp)
        self.weiss_green = fg.FrequencyGreen(self.green_sys.freq)
        self.time = np.linspace(self.parameters['time']['time_min'],
                                self.parameters['time']['time_max'],
                                self.parameters['time']['N_time'],)

    def impurity_solver(self) -> None:
        """Implementation of the iterated perturbation theory (IPT) solver.
        The self-energy is calculated using the weiss Green's function for the
        second order diagram and the full Green's function for the hartree
        part.
        """
        try:
            if self.first_iter is True:
                self.first_iter = False
        except AttributeError:
            self.first_iter = True
# ------------------ define frequency weiss green's function ---------------- #
        U = self.parameters['system']['U']
        epsilon = self.parameters['system']["e0"]

        self.weiss_green.dyson(self_energy=(
            self.hyb_dmft + self.hyb_leads), e_tot=epsilon)
# -------------------- get greater and lesser from keldysh ------------------ #
        if self.keldysh_comp == "keldysh":
            weiss_lesser = conv.get_lesser_from_keldysh(self.weiss_green)
            weiss_greater = conv.get_greater_from_keldysh(self.weiss_green)
        elif self.keldysh_comp == "lesser":
            weiss_lesser = self.weiss_green.keldysh
            weiss_greater = conv.get_greater_from_lesser(self.weiss_green)
# -------------------------------- time domain ------------------------------ #
        weiss_lesser_time = dft.dft(
            weiss_lesser, self.green_sys.freq, self.time, sign=-1)
        weiss_greater_time = dft.dft(
            weiss_greater, self.green_sys.freq, self.time, sign=-1)
# ---------------------------calculate self-energy -------------------------- #
        sigma_lesser_time = -1. * U * U * weiss_lesser_time *\
            weiss_greater_time.conj() * weiss_lesser_time
        sigma_greater_time = -1. * U * U * weiss_greater_time *\
            weiss_lesser_time.conj() * weiss_greater_time
        # If I directly fourier transform the retarded component, a factor of
        # 2 pi bigger then when calculated from the lesser and greater Green's
        # function. Therefore when multiplying that to the retarded component
        # before transforming in to the frequency domain the self consistency
        # converges and the occupation converges to 0.5 as it should
        sigma_retarded_time = 2 * np.pi * np.array([du.heaviside(t, 0) * (
            sigma_greater_time[i] - sigma_lesser_time[i])
            for i, t in enumerate(self.time)])
# ---------------------------- frequency domain ---------------------------- #
        sigma_lesser_freq = dft.dft(
            sigma_lesser_time, self.time, self.green_sys.freq, sign=1)
        sigma_retarded_freq = dft.dft(
            sigma_retarded_time, self.time, self.green_sys.freq, sign=1)
        sigma = fg.FrequencyGreen(
            self.green_sys.freq, retarded=sigma_retarded_freq,
            keldysh=sigma_lesser_freq)
        if self.keldysh_comp == "keldysh":
            self.self_energy_int = fg.FrequencyGreen(
                sigma.freq, retarded=sigma.retarded,
                keldysh=conv.get_keldysh_from_lesser(sigma))
        elif self.keldysh_comp == "lesser":
            self.self_energy_int = sigma
# ------------------------- update Green's function ------------------------- #

        self.green_sys.dyson(
            self_energy=self.self_energy_int + self.hyb_dmft + self.hyb_leads,
            e_tot=epsilon + U * (self.n - 0.5))

        return ()

    def solve(self):
        self.__solve__()


if __name__ == "__main__":
    #  Frequency grid
    N_grid = 2001
    freq_max = 10
    time_max = 20
    selfconsist_param = {'max_iter': 100, 'err_tol': 1e-6, 'mixing': 0.3}

    e0 = 0
    mu = 0
    beta = 10
    D = 30.1
    gamma = 0.02

    leads_param = {'e0': e0, 'mu': [mu], 'beta': beta, 'D': D, 'gamma': gamma}

    spinless = False
    spin_sector_max = 1
    tilde_conjugationrule_phase = True
    errors = []
    Us = [0, 3, 5, 7, 9]
    for U in Us:
        v = 1.0
        sys_param = {"e0": 0, 'v': v, 'U': U, 'spinless': spinless,
                     'tilde_conjugation': tilde_conjugationrule_phase}

        # Parameters of the auxiliary system
        Nb = 1
        nsite = 2 * Nb + 1
        aux_param = {'Nb': Nb, 'nsite': nsite}

        params = {'freq': {"freq_min": -freq_max, "freq_max": freq_max,
                           'N_freq': N_grid},
                  'time': {"time_min": -time_max, "time_max": time_max,
                           "N_time": N_grid},
                  'selfconsistency': selfconsist_param, 'leads': leads_param,
                  'aux_sys': aux_param, 'system': sys_param}

        # ################### Initializing Lindblad class ####################

        dmft_ipt = DMFT_IPT(params, hyb_leads=None, keldysh_comp="lesser")
        dmft_ipt.hyb_leads = dmft_ipt.get_bath()
        dmft_ipt.solve()
        plt.plot(dmft_ipt.green_sys.freq, -(1 / np.pi)
                 * dmft_ipt.green_sys.retarded.imag, label=U)
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$A(\omega)$")
        plt.legend()
        errors.append(dmft_ipt.err_iterations)
    plt.show()
    for i, U in enumerate(Us):
        plt.plot(errors[i], label=U)
        plt.yscale("log")
        plt.xlabel(r"Iteration")
        plt.ylabel(r"$||G_{new}(\omega)-G_{old}(\omega)||$")
        plt.legend()
    plt.show()
# %%
