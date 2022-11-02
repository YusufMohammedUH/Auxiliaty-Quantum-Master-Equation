from typing import Dict, Union
import numpy as np
import src.dmft.dmft_base as dmft
import src.greens_function.frequency_greens_function as fg
import src.greens_function.dos_util as du
import src.util.fourier as dft


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
        self.weiss_green = fg.FrequencyGreen(self.green_sys.freq,
                                             keldysh_comp=keldysh_comp)
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
        weiss_lesser = self.weiss_green.get_lesser()
        weiss_greater = self.weiss_green.get_greater()
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
            keldysh=sigma_lesser_freq, keldysh_comp='lesser')
        if self.keldysh_comp == "keldysh":
            self.self_energy_int = fg.FrequencyGreen(
                sigma.freq, retarded=sigma.retarded,
                keldysh=sigma.get_keldysh(), keldysh_comp='keldysh')
        elif self.keldysh_comp == "lesser":
            self.self_energy_int = sigma
# ------------------------- update Green's function ------------------------- #

        self.green_sys.dyson(
            self_energy=self.self_energy_int + self.hyb_dmft + self.hyb_leads,
            e_tot=epsilon + U * (self.n - 0.5))

        return ()

    def solve(self):
        self.__solve__()
