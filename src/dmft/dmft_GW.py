# %%
from typing import Dict, Union
import numpy as np
import src.dmft.dmft_base as dmft_base
import src.greens_function.frequency_greens_function as fg
import src.greens_function.dos_util as du
import src.util.fourier as dft
import matplotlib.pyplot as plt
# TODO: 0. calculate the Hartree part of the self-energy
#       1. calculate P(1;2)= -iG(1;2)G(2;1) (Fourier transform time)
#       2. calculate W(w) = U/[1-U*P(w)]    (Fourier transform frequency)
#       3. calculate Self-energy S_GW=iG(1;2)W(1;2) (Fourier transform time)
#       3. calculate green's function   (Fourier transform frequency)
#                   G(1;2) = [G_0^{-1}(1;2) -[S_Hartree + S_GW(1;2)]]^{-1}


class DMFT_GW(dmft_base.DMFTBase):
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
        self.screened_interaction = fg.FrequencyGreen(
            self.green_sys.freq, fermionic=False, keldysh_comp=keldysh_comp)
        self.polarizability = fg.FrequencyGreen(
            self.green_sys.freq, fermionic=False, keldysh_comp=keldysh_comp)
        self.time = np.linspace(self.parameters['time']['time_min'],
                                self.parameters['time']['time_max'],
                                self.parameters['time']['N_time'],)

    def impurity_solver(self) -> None:
        """Implementation of the iterated perturbation theory (IPT) solver.
        The self-energy is calculated using the weiss Green's function for the
        second order diagram and the full Green's function for the hartree
        part.

        """
# ------------------------- Calculate polarizability ------------------------ #
        U = self.parameters['system']['U']
        if U != 0:
            green_lesser_freq = self.green_sys.get_lesser()
            green_greater_freq = self.green_sys.get_greater()

            green_lesser_time = \
                dft.dft(green_lesser_freq,
                        self.green_sys.freq, self.time, sign=-1)
            green_greater_time = \
                dft.dft(green_greater_freq,
                        self.green_sys.freq, self.time, sign=-1)
            green_retarded_time = 2 * np.pi * np.array(
                [du.heaviside(t, 0) * (
                    green_greater_time[i] - green_lesser_time[i])
                    for i, t in enumerate(self.time)])

            polarizability_lesser_time = 1j * green_lesser_time * \
                green_greater_time.conj()

            polarizability_retarded_time = 1j * green_retarded_time \
                * green_lesser_time.conj() \
                - 1j * green_lesser_time * green_retarded_time.conj()

            self.polarizability = fg.FrequencyGreen(
                freq=self.green_sys.freq,
                retarded=dft.dft(polarizability_retarded_time, self.time,
                                 self.green_sys.freq, sign=1),
                keldysh=dft.dft(polarizability_lesser_time,
                                self.time, self.green_sys.freq, sign=1),
                fermionic=False, keldysh_comp=self.keldysh_comp)
# ---------------------------- screened interaction ------------------------- #
            polarizability_tmp = fg.FrequencyGreen(
                freq=self.polarizability.freq,
                retarded=self.polarizability.retarded * U,
                keldysh=self.polarizability.keldysh,
                fermionic=False,
                keldysh_comp=self.keldysh_comp)

            self.screened_interaction.dyson(
                self_energy=polarizability_tmp,
                g0_inv=1.)
            self.screened_interaction.retarded *= U
            self.screened_interaction.keldysh *= U**2
# ----------------------------- get self energy ----------------------------- #
            screened_int_lesser_freq = self.screened_interaction.get_lesser()
            screened_int_greater_freq = self.screened_interaction.get_greater()

            # fourier to time domain
            screened_int_lesser_time = dft.dft(screened_int_lesser_freq,
                                               self.green_sys.freq, self.time,
                                               sign=-1)

            screened_int_greater_time = dft.dft(screened_int_greater_freq,
                                                self.green_sys.freq, self.time,
                                                sign=-1)
            # XXX: self-consistency is conserving only while using
            #     \Sigma^{R} = \Theta(t)[\Sigma^{>}-\Sigma^{<}]
            #     at the moment I don't know why the equation form the paper
            #     doesn't work

            # screened_int_retarded_time = 2 * np.pi * np.array(
            #     [du.heaviside(t, 0) * (
            #         screened_int_greater_time[i] - screened_int_lesser_time[i])
            #         for i, t in enumerate(self.time)])

            # calculate self-ernergy in time
            sigma_lesser_time = 1j * green_lesser_time \
                * screened_int_lesser_time
            sigma_greater_time = 1j * green_greater_time \
                * screened_int_greater_time
            sigma_retarded_time2 = (2 * np.pi) * np.array(
                [du.heaviside(t, 0) * (
                    sigma_greater_time[i] - sigma_lesser_time[i])
                    for i, t in enumerate(self.time)])
            # sigma_retarded_time = (1j / 1.) * (
            #     green_retarded_time * screened_int_greater_time
            #     + green_lesser_freq * screened_int_retarded_time)

            # return to freqency domain
            sigma_lesser_freq = dft.dft(
                sigma_lesser_time, self.time, self.green_sys.freq, sign=1)
            sigma_retarded_freq = dft.dft(
                sigma_retarded_time2, self.time, self.green_sys.freq, sign=1)
            sigma = fg.FrequencyGreen(
                self.green_sys.freq, retarded=sigma_retarded_freq,
                keldysh=sigma_lesser_freq)
            if self.keldysh_comp == "keldysh":
                self.self_energy_int = fg.FrequencyGreen(
                    sigma.freq, retarded=sigma.retarded,
                    keldysh=sigma.get_keldysh())
            elif self.keldysh_comp == "lesser":
                self.self_energy_int = sigma

# ------------------------- update Green's function ------------------------- #
        epsilon = self.parameters['system']["e0"]
        self.green_sys.dyson(
            self_energy=self.self_energy_int + self.hyb_dmft + self.hyb_leads,
            e_tot=epsilon + U * (self.n - 0.5))
        return ()

    def solve(self):
        self.__solve__()


if __name__ == "__main__":
    #  Frequency grid
    N_grid = 2001
    freq_max = 30
    time_max = 20
    selfconsist_param = {'max_iter': 50, 'err_tol': 1e-6, 'mixing': 0.3}

    e0 = 0
    mu = 0
    beta = 10
    D = 30.1
    gamma = 0.1

    leads_param = {'e0': e0, 'mu': [mu], 'beta': beta, 'D': D, 'gamma': gamma}

    spinless = False
    spin_sector_max = 1
    tilde_conjugationrule_phase = True
    errors = []
    Us = [4]  # [0, 1, 2, 3, 4]
    for U in Us:
        print("U: ", U)
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

        # ##################### Initializing Lindblad class #######################

        dmft_gw = DMFT_GW(params, hyb_leads=None, keldysh_comp="lesser")
        dmft_gw.hyb_leads = dmft_gw.get_bath()
        dmft_gw.solve()
        plt.plot(dmft_gw.green_sys.freq, -(1 / np.pi)
                 * dmft_gw.green_sys.retarded.imag, label=U)
        plt.xlabel(r"$\omega$")
        plt.ylabel(r"$A(\omega)$")
        plt.legend()
        errors.append(dmft_gw.err_iterations)
    plt.show()
    for i, U in enumerate(Us):
        plt.plot(errors[i], label=U)
        plt.yscale("log")
        plt.xlabel(r"Iteration")
        plt.ylabel(r"$||G_{new}(\omega)-G_{old}(\omega)||$")
        plt.legend()
    plt.show()

# %%
