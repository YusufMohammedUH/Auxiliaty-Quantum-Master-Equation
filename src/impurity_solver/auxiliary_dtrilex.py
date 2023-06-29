# %%
from typing import Dict, Optional, Tuple
from numba import njit
import numpy as np
# from scipy.integrate import simps
import matplotlib.pyplot as plt
import src.greens_function.frequency_greens_function as fg
import src.greens_function.correlation_functions as corr
import src.util.hdf5_util as hd5
# import src.greens_function.dos_util as du
# import src.util.fourier as dft
import src.impurity_solver.auxiliary_solver_base as aux_base
# import src.dmft.auxiliary_dmft as aux_dmft

# import src.super_fermionic_space.super_fermionic_subspace as sf_sub
# import src.super_fermionic_space.model_lindbladian as lind
# Dual TRILEX:
#  [X] 1. inherit from AuxiliaryDualSolverBase:
#  [X] 2. copy from AuxiliaryDualTRILEX
#  [X] 3. calculate the three point correlator
#  [X] 4. calculate convolution of three point correlator with green's function
#         and obtain three point vertex
#  [X] 5. calculate "mirrored" three point vertex
#  [X] 6. calculate polarization:
#           [X]- first derive convolution for the polarization
#           [X]- then calculate the polarization
#  [X] 7. calculate self-energy
#           [X]- first derive convolution for the self-energy
#           [X]- then calculate the self-energy


@njit(cache=True)
def convolute_polarization_contour_component(omega_idx: int,
                                             gamma: np.ndarray,
                                             gamma_tilde: np.ndarray,
                                             green_1: np.ndarray,
                                             green_2: np.ndarray
                                             ) -> np.complex128:
    """Convolute the vertices and green's functions for fixed contour
    components and fixed frequency.

    Parameters
    ----------
    omega_idx : int
        Index of the target frequency

    gamma : np.ndarray
        Three point vertex

    gamma_tilde : np.ndarray
        Three point vertex mirrored

    green_1 : np.ndarray
        Green's function

    green_2 : np.ndarray
        Green's function

    Returns
    -------
    np.complex128
        Value of the convolution at index omega_idx
    """
    N = len(green_1)
    polarisation_contour_comp = np.complex128(0)
    s = N // 2 - omega_idx
    for ii in range(-s, N - s + 1):
        if ii < 0:
            ii = 0
        elif ii > N:
            ii = N

        polarisation_contour_comp += gamma[ii, ii - omega_idx] * \
            gamma_tilde[omega_idx - ii, -ii] * green_1[ii] *\
            green_2[omega_idx - ii]

    return polarisation_contour_comp


@njit(parallel=True, cache=True)
def get_polarization_contour_component(contour_comp: Tuple[int, int, int, int],
                                       gamma: np.ndarray,
                                       gamma_tilde: np.ndarray,
                                       green_1: np.ndarray,
                                       green_2: np.ndarray,
                                       contour_signs: np.ndarray
                                       ) -> np.complex128:
    """Obtain polarization by convoluting the vertices and green's functions
    for fixed contour components.

    Parameters
    ----------
    gamma : np.ndarray
        Three point vertex

    gamma_tilde : np.ndarray
        Three point vertex mirrored

    green_1 : np.ndarray
        Green's function

    green_2 : np.ndarray
        Green's function

    Returns
    -------
    np.ndarray
        Resulting function
    """
    N = len(green_1)
    polarisation_contour_comp = np.zeros(N, dtype=np.complex128)
    sign = 1
    for contour in contour_comp:
        sign *= contour_signs[contour]
    for ii in range(N):
        polarisation_contour_comp[ii] = (-1j / (2 * np.pi)) * sign * \
            convolute_polarization_contour_component(ii, gamma, gamma_tilde,
                                                     green_1, green_2)


@njit(parallel=True, cache=True)
def get_self_energy_contour_component(contour_comp: Tuple[int, int, int, int],
                                      gamma: np.ndarray,
                                      gamma_tilde: np.ndarray,
                                      green_1: np.ndarray,
                                      green_2: np.ndarray,
                                      contour_signs: np.ndarray
                                      ) -> np.complex128:
    """Convolute the vertices and green's functions for fixed contour
    components.

    Parameters
    ----------
    gamma : np.ndarray
        Three point vertex

    gamma_tilde : np.ndarray
        Three point vertex mirrored

    green_1 : np.ndarray
        Green's function

    green_2 : np.ndarray
        Green's function

    Returns
    -------
    np.ndarray
        Resulting function
    """
    N = len(green_1)
    self_energy_contour_comp = np.zeros(N, dtype=np.complex128)
    sign = 1
    for contour in contour_comp:
        sign *= contour_signs[contour]
    for ii in range(N):
        for jj in range(N):
            if ii + (N - 1 - jj) < 0:
                self_energy_contour_comp[ii] += np.complex128(0)
            elif ii + (N - 1 - jj) > N:
                self_energy_contour_comp[ii] += np.complex128(0)
            else:
                self_energy_contour_comp[ii] += (1j / (2 * np.pi)) * sign * \
                    gamma[jj, N - 1 - ii] * gamma_tilde[ii,
                                                        N - 1 - jj] *\
                    green_1[ii + (N - 1 - jj)] * green_2[jj]


class AuxiliaryDualTRILEX(aux_base.AuxiliaryDualSolverBase):
    """This class facilitates the calculation of the system Green's function
    by using a auxiliary system as a starting point for steady state dual
    TRILEX


    Parameters
    ----------
    filename : Optional[str], optional
        File name in which data is stored, by default None

    dir_ : Optional[str], optional
        HDF5 group/ subdirectory in which data is stored, by default None

    dataname : Optional[str], optional
        Name under which data is stored, by default None

    load_input_param : bool, optional
        Load input parameters like green_aux, hyb_sys, etc. if True, by default
        True.

    load_aux_data : bool, optional
        Load auxiliary objects calculated here like auxiliary polarization
        etc. if True, by default False

    U_trilex : Dict
        Interaction strength for the trilex channels

    green_aux : fg.FrequencyGreen
        Auxiliary Green's function

    hyb_sys : fg.FrequencyGreen
        System hybridization function

    hyb_aux : fg.FrequencyGreen
        Auxiliary hybridization function

    correlators : corr.Correlators
        Correlators object used to calculate the vertices.

    Attributes
    ----------
    U_trilex : Dict
        Interaction strength for the trilex channels

    green_aux : fg.FrequencyGreen
        Auxiliary Green's function

    hyb_sys : fg.FrequencyGreen
        System hybridization function

    hyb_aux : fg.FrequencyGreen
        Auxiliary hybridization function

    delta_hyb : fg.FrequencyGreen
        Difference between the auxiliary and system hybridization function

    green_sys : fg.FrequencyGreen
        System Green's function

    green_bare_dual : fg.FrequencyGreen
        Bare fermionic dual Green's function

    green_dual : fg.FrequencyGreen
        Fermionic dual Green's function

    sigma_dual : fg.FrequencyGreen
        Dual fermionic self-energy

    bare_dual_screened_interaction : Dict
        Bare bosonic dual Green's function

    green_dual_boson : Dict
        Bosonic dual Green's function

    polarization_aux : Dict
        Auxiliary polarization

    susceptibility_aux : Dict
        Auxiliary susceptibility

    correlators : corr.Correlators
        Correlators object used to calculate the vertices.

    Raises
    ------
    ValueError
        'Either all or none of the following arguments must be given: filename,
        dir_, dataname. If nonn are given, U_trilex, green_aux, hyb_sys,
        hyb_aux and correlators are required.'
    """

    def __init__(self, *, filename: Optional[str] = None,
                 dir_: Optional[str] = None,
                 dataname: Optional[str] = None,
                 load_input_param: bool = True,
                 load_aux_data: bool = False,
                 time_param: Optional[Dict] = None,
                 U_trilex: Optional[Dict] = None,
                 green_aux: Optional[fg.FrequencyGreen] = None,
                 hyb_sys: Optional[fg.FrequencyGreen] = None,
                 hyb_aux: Optional[fg.FrequencyGreen] = None,
                 correlators: Optional[corr.Correlators] = None,
                 keldysh_comp: str = 'keldysh') -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.time = np.linspace(time_param['time_min'],
                                time_param['time_max'],
                                time_param['N_time'])

        self.contour_components = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

        self.three_point_correlator = {('up', 'up', 'ch'): None,
                                       ('up', 'do', 'x'): None,
                                       ('up', 'do', 'y'): None,
                                       ('up', 'up', 'z'): None}

        self.three_point_vertex = {('up', 'up', 'ch'): None,
                                   ('up', 'do', 'x'): None,
                                   ('up', 'do', 'y'): None,
                                   ('up', 'up', 'z'): None}
        # self.four_point_vertex = {}

        super().__init__(filename=filename, dir_=dir_, dataname=dataname,
                         load_input_param=load_input_param,
                         load_aux_data=load_aux_data,
                         U_trilex=U_trilex,
                         green_aux=green_aux,
                         hyb_sys=hyb_sys,
                         hyb_aux=hyb_aux,
                         correlators=correlators,
                         keldysh_comp=keldysh_comp)

        # acording to Zhenya's notes
        # self.polarization_dual = {key1: {
        #     key2: fg.FrequencyGreen(freq=self.hyb_aux.freq,
        #                             fermionic=False,
        #                             keldysh_comp=self.keldysh_comp,
        #                             orbitals=green_aux.orbitals)
        #     for key2 in self.three_point_vertex}
        #     for key1 in self.bare_dual_screened_interaction}

    def compute_three_point_correlator(self) -> None:
        """Calculate the three point vertex
        """
        for spins in self.three_point_correlator:
            self.three_point_correlator[spins] = \
                self.correlators.get_three_point_vertex(
                freq=self.green_aux.freq, spin=spins, return_=True)

    def compute_three_point_vertex(self) -> None:
        """Calculate the three point vertex, by convoluting the green's
        functions and susceptibilities.
        """
        for spins in self.three_point_vertex:
            for con_comp in self.contour_components:
                self.three_point_vertex[spins][con_comp] = \
                    self.correlators.get_vertex_green_convolution(
                        green=self.green_aux.inverse(), component=con_comp,
                        spin=spins, position_freq=(0, 0),
                        vertex=self.three_point_correlator)

                self.three_point_vertex[spins][con_comp] = \
                    self.correlators.get_vertex_green_convolution(
                        green=self.green_aux.inverse(), component=con_comp,
                        spin=spins, position_freq=(1, 1),
                        vertex=self.three_point_vertex)

                # XXX: function passed should be different from susceptibility
                self.three_point_vertex[spins][con_comp] = \
                    self.correlators.get_vertex_green_convolution(
                        green=self.susceptibility_aux.inverse(),
                        component=con_comp,
                        spin=spins, position_freq=(1, 2),
                        vertex=self.three_point_vertex)

    def get_conjugated_contour_component(self, component: tuple) -> tuple:

        return (1 - component[0], 1 - component[1], 1 - component[2])

    def get_mirrored_three_point_vertex(self, component: tuple, spin: tuple
                                        ) -> None:
        """obtain the mirrored three point vertex

        Parameters
        ----------
        component : tuple
            contour component of the vertex,e.g. (0, 0, 0) is for all times on
            the upper contour/ forward time branch

        spin : tuple
            tuple of spins, e.g. ('up', 'up', 'ch') is for spin up creation and
            annihilation operator and charge channel operator

        Returns
        -------
        _type_
            _description_
        """
        # Here we simply implement the symmetry relations written in the
        # auxiliary dual boson paper of Feng
        conj_component = (1 - component[0], 1 - component[1], 1 - component[2])
        return self.three_point_vertex[spin][conj_component].T.conj()

    def compute_polarization_dual(self, green: "fg.FrequencyGreen" = None
                                  ) -> None:
        """Calculate the dual polarization for given green's function and
        supplied branches of the contour.

        Parameters
        ----------
        green : fg.FrequencyGreen, optional
            Single particle Green's function, by default None
        """
        if green is None:
            green = self.green_dual
        self.compute_three_point_correlator()
        self.compute_three_point_vertex()
        polarization_dual_contour_comp = np.zeros((2, 2,
                                                   len(self.green_aux.freq)),
                                                  np.complex128)
        contour_sign = [1, -1]
        # Following PRB 100 205115 (2019) both vertices are in the same channel
        # while Zhenya's notes introduce a screened interaction with two
        # channel indices. which would lead to the assumption that the
        # polarization could include two different vertices.
        for spins in self.three_point_vertex.keys():
            for c1 in [0, 1]:
                for c2 in [0, 1]:
                    for c3 in [0, 1]:
                        for c4 in [0, 1]:
                            for c5 in [0, 1]:
                                for c6 in [0, 1]:
                                    contour_comp = (c3, c4, c5, c6)
                                    gamma_cont_conj = \
                                        self.get_conjugated_contour_component(
                                            (c1, c4, c5))
                                    gamma = \
                                        self.get_mirrored_three_point_vertex(
                                            gamma_cont_conj, spins)
                                    gamma_tilde = \
                                        self.three_point_vertex[spins][
                                            c6, c3, c2]
                                    green_1 = green.get_countour_greens(
                                        c3, c4)
                                    green_2 = green.get_countour_greens(
                                        c5, c6)
                                    polarization_dual_contour_comp[c1, c2] += \
                                        get_polarization_contour_component(
                                            contour_comp=contour_comp,
                                            gamma=gamma,
                                            gamma_tilde=gamma_tilde,
                                            green_1=green_1,
                                            green_2=green_2,
                                            contour_signs=contour_sign)

            self.polarization_dual[spins].retarded = (
                polarization_dual_contour_comp[0, 0]
                - polarization_dual_contour_comp[0, 1]
                + polarization_dual_contour_comp[1, 0]
                - polarization_dual_contour_comp[1, 1]) * 0.5

            if self.polarization_dual[spins].keldysh_comp == 'lesser':
                self.polarization_dual[spins].keldysh = \
                    polarization_dual_contour_comp[0, 1]
            elif self.polarization_dual[spins].keldysh_comp == 'greater':
                self.polarization_dual[spins].keldysh = \
                    polarization_dual_contour_comp[1, 0]
            elif self.polarization_dual[spins].keldysh_comp == 'keldysh':
                self.polarization_dual[spins].keldysh = -1j * \
                    self.polarization_dual[spins].get_spectral_function() \
                    + 2. * polarization_dual_contour_comp[0, 1]

    def compute_sigma_dual(self, green: "fg.FrequencyGreen" = None) -> None:
        """Calculate the dual self energy for given green's function and
        supplied branches of the contour.

        Parameters
        ----------
        green : fg.FrequencyGreen, optional
            Single particle Green's function, by default None
        """
        if green is None:
            green = self.green_dual

        sigma_dual_temp = np.zeros((2, 2, len(self.green_aux.freq)),
                                   np.complex128)
        contour_sign = [1, -1]

        for spins in self.three_point_vertex.keys():
            for c1 in [0, 1]:
                for c2 in [0, 1]:
                    for c3 in [0, 1]:
                        for c4 in [0, 1]:
                            for c5 in [0, 1]:
                                for c6 in [0, 1]:
                                    contour_comp = (c3, c4, c5, c6)
                                    gamma_cont_conj = \
                                        self.get_conjugated_contour_component(
                                            (c5, c3, c2))
                                    gamma = \
                                        self.get_mirrored_three_point_vertex(
                                            gamma_cont_conj, spins)
                                    gamma_tilde = \
                                        self.three_point_vertex[spins][
                                            c1, c4, c6]
                                    green_1 = self.dual_screened_interaction[
                                        spins].get_countour_greens(c6, c5)
                                    green_2 = green.get_countour_greens(
                                        c4, c3)
                                    sigma_dual_temp[c1, c2] += \
                                        get_polarization_contour_component(
                                            contour_comp=contour_comp,
                                            gamma=gamma,
                                            gamma_tilde=gamma_tilde,
                                            green_1=green_1,
                                            green_2=green_2,
                                            contour_signs=contour_sign)

        self.sigma_dual.retarded = (sigma_dual_temp[0, 0]
                                    - sigma_dual_temp[0, 1]
                                    + sigma_dual_temp[1, 0]
                                    - sigma_dual_temp[1, 1]) * 0.5

        if self.sigma_dual.keldysh_comp == 'lesser':
            self.sigma_dual.keldysh = sigma_dual_temp[0, 1]
        elif self.sigma_dual.keldysh_comp == 'greater':
            self.sigma_dual.keldysh = sigma_dual_temp[1, 0]
        elif self.sigma_dual.keldysh_comp == 'keldysh':
            self.sigma_dual.keldysh = -1j \
                * self.sigma_dual.get_spectral_function() \
                + 2. * sigma_dual_temp[0, 1]

    def solve(self, err_tol=1e-7, iter_max=100):
        self.__solve__(err_tol=err_tol, iter_max=iter_max)

    def save(self, fname: str, dir_: str, dataname: str,
             save_input_param: bool = True, save_aux_data: bool = False
             ) -> None:
        self.__save__(fname=fname, dir_=dir_, dataname=dataname,
                      save_input_param=save_input_param,
                      save_aux_data=save_aux_data)

        freq = {'freq_min': self.hyb_aux.freq[0],
                'freq_max': self.hyb_aux.freq[-1],
                'N_freq': len(self.hyb_aux.freq)}

        time = {'time_min': self.time[0],
                'time_max': self.time[-1],
                'N_time': len(self.time)}
        hd5.add_attrs(fname, "/", {'freq': freq, 'time': time})

        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'three_point_vertex',
                          self.three_point_vertex)
        hd5.add_dict_data(fname, f"{dir_}/{dataname}", 'four_point_vertex',
                          self.four_point_vertex)

    def load(self, fname: str, dir_: str, dataname: str,
             load_input_param: bool = True, load_aux_data: bool = False
             ) -> None:
        print('bla')
        self.__load__(fname=fname, dir_=dir_, dataname=dataname,
                      load_input_param=load_input_param,
                      load_aux_data=load_aux_data)

        if load_input_param:

            grid = hd5.read_attrs(fname, '/')
            try:
                self.time = np.linspace(
                    grid['time']['time_min'], grid['time']['time_max'],
                    grid['time']['N_time'])
            except KeyError:
                print(KeyError)
        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'three_point_vertex')
        for key in self.three_point_vertex:
            self.three_point_vertex[key] = temp[f"{key}"]

        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'four_point_vertex')
        # for key in self.four_point_vertex.keys():
        #     self.four_point_vertex[key] = temp[f"{key}"]


# %%
if __name__ == '__main__':
    import colorcet as cc

    time_param = {'time_min': -10, 'time_max': 10, 'N_time': 1001}
    auxTrilex = AuxiliaryDualTRILEX(
        filename='ForAuxTrilex.h5', dir_='', dataname='trilex',
        time_param=time_param)

    plt.imshow(auxTrilex.three_point_vertex[
        ('up', 'do', 'x')][:, :, 0, 0, 0].imag, cmap=cc.cm.bgy)
    plt.colorbar()
    plt.show()

# %%
