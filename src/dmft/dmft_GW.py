from typing import Dict, Union
import numpy as np
import src.dmft.dmft_base as dmft_base
import src.greens_function.frequency_greens_function as fg
import src.greens_function.dos_util as du
import src.util.fourier as dft


class DMFT_GW(dmft_base.DMFTBase):
    """Implementation of DMFT with the iterated perturbation theory (IPT)
    solver.

    Parameters
    ----------
    dmft.DMFTBase : dmft.DMFTBase
        Parent class in which the solver implemented here is used to solve
        DMFT calculations.
    """

    def __init__(self, parameters: Union[Dict, None] = None,
                 hyb_leads: Union[fg.FrequencyGreen, None] = None,
                 fname: Union[str, None] = None) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        super().__init__(parameters=parameters, hyb_leads=hyb_leads,
                         fname=fname)
        self.screened_interaction = fg.FrequencyGreen(
            self.green_sys.freq, fermionic=False,
            keldysh_comp=self.keldysh_comp)
        self.polarizability = fg.FrequencyGreen(
            self.green_sys.freq, fermionic=False,
            keldysh_comp=self.keldysh_comp)
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
            #         screened_int_greater_time[i]
            # - screened_int_lesser_time[i])
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
