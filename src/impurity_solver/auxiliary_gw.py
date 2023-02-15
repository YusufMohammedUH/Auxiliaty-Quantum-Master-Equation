# %%
from typing import Dict, Optional
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import src.super_fermionic_space.model_lindbladian as lind
import src.greens_function.frequency_greens_function as fg
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.greens_function.correlation_functions as corr
import src.auxiliary_mapping.optimization_auxiliary_hybridization as opt
import src.util.hdf5_util as hd5
import src.greens_function.dos_util as du
import src.util.fourier as dft
import src.dmft.auxiliary_dmft as aux_dmft
# Dual GW:
#  [X] 0. get g_aux, hyb_sys, hyb_aux
#  [X] 1. calculate auxiliary susceptibility
#  [X] 2. calculate auxiliary polarization
#  [X] 3. calculate non-interacting dual Green's function and bosonic
#         propagator
#
#  [X] 5. calculate dual polarization
#  [X] 7. calculate the dressed dual bosonic propagator
#  [X] 8. calculate dual self-energy
#  [X] 9. calculate the dressed dual Green's function
#   (optional) update self-consistently 6. with dressed dual Green's function
#              until convergence
#  [X] 10. calculate the system Green's function


class AuxiliaryDualGW:
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

    three_point_vertex : fg.FrequencyGreen
        Dual four-point vertex

    four_point_vertex : fg.FrequencyGreen
        Dual four-point vertex

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
                 time_param: Optional[np.ndarray] = None,
                 U_trilex: Optional[Dict] = None,
                 green_aux: Optional[fg.FrequencyGreen] = None,
                 hyb_sys: Optional[fg.FrequencyGreen] = None,
                 hyb_aux: Optional[fg.FrequencyGreen] = None,
                 correlators: Optional[corr.Correlators] = None,
                 keldysh_comp: str = 'keldysh') -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """

        self.correlators = correlators
        if (filename is None) and (dir_ is None) and (dataname is None):
            if hyb_aux.keldysh_comp != green_aux.keldysh_comp or \
                    hyb_aux.keldysh_comp != hyb_sys.keldysh_comp:
                raise ValueError("All Green's functions must have the same "
                                 "keldysh component.")

            self.time = np.linspace(time_param['time_min'],
                                    time_param['time_max'],
                                    time_param['N_time'])
            self.U_trilex = U_trilex
            self.keldysh_comp = keldysh_comp
            self.err_iterations_gw = []

            if self.keldysh_comp == green_aux.keldysh_comp:
                self.green_aux = green_aux
            else:
                if self.keldysh_comp == 'keldysh':
                    self.green_aux = fg.FrequencyGreen(
                        freq=green_aux.freq, retarded=green_aux.retarded,
                        keldysh=green_aux.get_keldysh(),
                        keldysh_comp=self.keldysh_comp,
                        orbitals=green_aux.orbitals)
                elif self.keldysh_comp == 'lesser':
                    self.green_aux = fg.FrequencyGreen(
                        freq=green_aux.freq, retarded=green_aux.retarded,
                        keldysh=green_aux.get_lesser(),
                        keldysh_comp=self.keldysh_comp,
                        orbitals=green_aux.orbitals)
            if self.keldysh_comp == hyb_sys.keldysh_comp:
                self.hyb_sys = hyb_sys
            else:
                if self.keldysh_comp == 'keldysh':
                    self.hyb_sys = fg.FrequencyGreen(
                        freq=hyb_sys.freq, retarded=hyb_sys.retarded,
                        keldysh=hyb_sys.get_keldysh(),
                        keldysh_comp=self.keldysh_comp,
                        orbitals=hyb_sys.orbitals)
                elif self.keldysh_comp == 'lesser':
                    self.hyb_sys = fg.FrequencyGreen(
                        freq=hyb_sys.freq, retarded=hyb_sys.retarded,
                        keldysh=hyb_sys.get_lesser(),
                        keldysh_comp=self.keldysh_comp,
                        orbitals=hyb_sys.orbitals)
            if self.keldysh_comp == hyb_aux.keldysh_comp:
                self.hyb_aux = hyb_aux
            else:
                if self.keldysh_comp == 'keldysh':
                    self.hyb_aux = fg.FrequencyGreen(
                        freq=hyb_aux.freq, retarded=hyb_aux.retarded,
                        keldysh=hyb_aux.get_keldysh(),
                        keldysh_comp=self.keldysh_comp,
                        orbitals=hyb_aux.orbitals)
                elif self.keldysh_comp == 'lesser':
                    self.hyb_aux = fg.FrequencyGreen(
                        freq=hyb_aux.freq, retarded=hyb_aux.retarded,
                        keldysh=hyb_aux.get_lesser(),
                        keldysh_comp=self.keldysh_comp,
                        orbitals=hyb_aux.orbitals)

            self.delta_hyb = self.hyb_aux - self.hyb_sys
            self.sigma_hartree = None
            self.green_sys = fg.FrequencyGreen(
                freq=self.hyb_aux.freq,
                keldysh_comp=self.keldysh_comp,
                orbitals=green_aux.orbitals)

            self.green_bare_dual = fg.FrequencyGreen(
                freq=self.hyb_aux.freq,
                keldysh_comp=self.keldysh_comp,
                orbitals=green_aux.orbitals)
            self.green_dual = fg.FrequencyGreen(
                freq=self.hyb_aux.freq,
                keldysh_comp=self.keldysh_comp,
                orbitals=green_aux.orbitals)
            self.sigma_dual = fg.FrequencyGreen(
                freq=self.hyb_aux.freq,
                fermionic=True,
                keldysh_comp=self.keldysh_comp,
                orbitals=green_aux.orbitals)

            # For now only the following channels are implemented
            self.bare_dual_screened_interaction = {
                ('ch', 'ch'): fg.FrequencyGreen(
                    freq=self.hyb_aux.freq,
                    fermionic=False,
                    keldysh_comp=self.keldysh_comp,
                    orbitals=green_aux.orbitals),
                ('x', 'x'): fg.FrequencyGreen(
                    freq=self.hyb_aux.freq,
                    fermionic=False,
                    keldysh_comp=self.keldysh_comp,
                    orbitals=green_aux.orbitals),
                ('x', 'y'): fg.FrequencyGreen(
                    freq=self.hyb_aux.freq,
                    fermionic=False,
                    keldysh_comp=self.keldysh_comp,
                    orbitals=green_aux.orbitals),
                ('y', 'x'): fg.FrequencyGreen(
                    freq=self.hyb_aux.freq,
                    fermionic=False,
                    keldysh_comp=self.keldysh_comp,
                    orbitals=green_aux.orbitals),
                ('y', 'y'): fg.FrequencyGreen(
                    freq=self.hyb_aux.freq,
                    fermionic=False,
                    keldysh_comp=self.keldysh_comp,
                    orbitals=green_aux.orbitals),
                ('z', 'z'): fg.FrequencyGreen(
                    freq=self.hyb_aux.freq,
                    fermionic=False,
                    keldysh_comp=self.keldysh_comp,
                    orbitals=green_aux.orbitals)}
            # It seams as if (x,y) and -(y,x) are the same channels
            # In paramagnetic case (z,ch) = (ch,z) = 0

            self.dual_screened_interaction = {
                key: fg.FrequencyGreen(freq=self.hyb_aux.freq,
                                       fermionic=False,
                                       keldysh_comp=self.keldysh_comp,
                                       orbitals=green_aux.orbitals)
                for key in self.bare_dual_screened_interaction}
            self.polarization_aux = {
                key: fg.FrequencyGreen(freq=self.hyb_aux.freq,
                                       fermionic=False,
                                       keldysh_comp=self.keldysh_comp,
                                       orbitals=green_aux.orbitals)
                for key in self.bare_dual_screened_interaction}
            self.susceptibility_aux = {
                key: fg.FrequencyGreen(freq=self.hyb_aux.freq,
                                       fermionic=False,
                                       keldysh_comp=self.keldysh_comp,
                                       orbitals=green_aux.orbitals)
                for key in self.bare_dual_screened_interaction}

            self.polarization_dual = fg.FrequencyGreen(
                freq=self.hyb_aux.freq,
                fermionic=False,
                keldysh_comp=self.keldysh_comp,
                orbitals=green_aux.orbitals)

            self.three_point_vertex = {('up', 'up', 'ch'): None,
                                       ('up', 'do', 'x'): None,
                                       ('up', 'do', 'y'): None,
                                       ('up', 'up', 'z'): None}
            self.four_point_vertex = {}
        elif (filename is not None) and (
                dir_ is not None) and (dataname is not None):
            self.load(fname=filename, dir_=dir_, dataname=dataname,
                      load_input_param=load_input_param,
                      load_aux_data=load_aux_data)
        else:
            raise ValueError('Either all or none of the following arguments ' +
                             'must be given: filename, dir_, dataname. ' +
                             'If nonn are given, U_trilex, green_aux,' +
                             ' hyb_sys, hyb_aux and ' +
                             'correlators are required.')

    def compute_polarization_aux(self) -> None:
        """Calculate the bare bosonic dual Green's function.
        """
        # XXX: Why not just calculate the bubble diagram directly?
        #       -> how would i calculate the charde
        for channel in self.bare_dual_screened_interaction:
            tmp = \
                self.correlators.get_susceptibility_physical(
                    self.green_aux.freq, channels=channel)
            if self.keldysh_comp == 'keldysh':
                self.susceptibility_aux[channel] = tmp
            elif self.keldysh_comp == 'lesser':
                self.susceptibility_aux[channel].retarded = tmp.retarded
                self.susceptibility_aux[channel].keldysh = tmp.get_lesser()

            self.polarization_aux[channel] = self.susceptibility_aux[channel] \
                * (self.susceptibility_aux[channel]
                    * self.U_trilex[channel[0]] + 1.).inverse()

    def compute_green_bare_dual(self) -> None:
        """Calculate the bare fermionic dual Green's function.
        """
        self.green_bare_dual = self.green_aux \
            * (self.green_aux + self.delta_hyb.inverse()).inverse() \
            * self.green_aux * (-1)

    def compute_bare_dual_screened_interaction(self) -> None:
        """Calculate the bare bosonic dual Green's function.
        """
        for channel in self.bare_dual_screened_interaction:
            self.bare_dual_screened_interaction[channel] = (
                self.polarization_aux[channel]
                * self.U_trilex[channel[0]]) \
                * (self.susceptibility_aux[channel]
                   * self.U_trilex[channel[0]] +
                   0.5) * self.polarization_aux[channel]

    def compute_polarization_dual(self, green: "fg.FrequencyGreen" = None
                                  ) -> None:
        if green is None:
            green = self.green_dual
        green_lesser_freq = green.get_lesser()
        green_greater_freq = green.get_greater()

        green_lesser_time = \
            dft.dft(green_lesser_freq,
                    self.green_sys.freq, self.time, sign=-1)
        green_greater_time = \
            dft.dft(green_greater_freq,
                    self.green_sys.freq, self.time, sign=-1)

        polarizability_lesser_time = 1j * green_lesser_time * \
            green_greater_time.conj()
        polarizability_greater_time = 1j * green_greater_time * \
            green_lesser_time.conj()
        polarizability_retarded_time = 2 * np.pi * np.array(
            [du.heaviside(t, 0) * (
                polarizability_greater_time[i] - polarizability_lesser_time[i])
                for i, t in enumerate(self.time)])

        polarizability_tmp = fg.FrequencyGreen(
            freq=self.green_sys.freq,
            retarded=dft.dft(polarizability_retarded_time, self.time,
                             self.green_sys.freq, sign=1),
            keldysh=dft.dft(polarizability_lesser_time,
                            self.time, self.green_sys.freq, sign=1),
            fermionic=False, keldysh_comp=self.keldysh_comp)

        if self.keldysh_comp == 'lesser':
            self.polarization_dual = polarizability_tmp
        elif self.keldysh_comp == 'keldysh':
            self.polarization_dual.retarded = polarizability_tmp.retarded
            self.polarization_dual.keldysh = polarizability_tmp.get_keldysh()

    def compute_dual_screened_interaction(self) -> None:
        """Calculate the bare bosonic dual Green's function.
        """
        for channel in self.dual_screened_interaction:
            self.dual_screened_interaction[channel].dyson(
                self_energy=self.polarization_dual,
                g0=self.bare_dual_screened_interaction[channel])
            # self.dual_screened_interaction[channel] = \
            #     self.bare_dual_screened_interaction[channel] * (
            #     self.bare_dual_screened_interaction[channel]
            #     * self.polarization_dual * (-1) + 1.).inverse()

    def compute_sigma_dual(self, green: "fg.FrequencyGreen" = None) -> None:
        if green is None:
            green = self.green_dual

        green_dual_lesser = green.get_lesser()
        green_dual_greater = green.get_greater()

        green_dual_lesser_time = dft.dft(green_dual_lesser,
                                         self.green_sys.freq, self.time,
                                         sign=-1)
        green_dual_greater_time = dft.dft(green_dual_greater,
                                          self.green_sys.freq, self.time,
                                          sign=-1)
        self.sigma_hartree = 0
        sigma_dual_tmp = fg.FrequencyGreen(
            freq=self.green_aux.freq, keldysh_comp='lesser',
            orbitals=self.green_aux.orbitals)

        for channel in self.dual_screened_interaction:

            screened_dual_lesser = \
                self.dual_screened_interaction[channel].get_lesser()
            screened_dual_greater = \
                self.dual_screened_interaction[channel].get_greater()

            screened_dual_lesser_time = dft.dft(screened_dual_lesser,
                                                self.green_sys.freq, self.time,
                                                sign=-1)
            screened_dual_greater_time = dft.dft(screened_dual_greater,
                                                 self.green_sys.freq,
                                                 self.time,
                                                 sign=-1)
            screened_dual_retarded_time = (2 * np.pi) * np.array(
                [du.heaviside(t, 0) * (
                    screened_dual_greater_time[i]
                    - screened_dual_lesser_time[i])
                    for i, t in enumerate(self.time)])

            screened_dual_advanced_time = (2 * np.pi) * np.array(
                [du.heaviside(0, t) * (
                    screened_dual_greater_time[i]
                    - screened_dual_lesser_time[i])
                    for i, t in enumerate(self.time)])

            tmp_gw_1_lesser_time = 1j * green_dual_lesser_time \
                * screened_dual_lesser_time
            tmp_gw_1_greater_time = 1j * green_dual_greater_time \
                * screened_dual_greater_time
            tmp_gw_1_retarded_time = (2 * np.pi) * np.array(
                [du.heaviside(t, 0) * (
                    tmp_gw_1_greater_time[i] - tmp_gw_1_lesser_time[i])
                    for i, t in enumerate(self.time)])

            tmp_gw_2_lesser_time = 1j * green_dual_lesser_time \
                * screened_dual_greater_time
            tmp_gw_2_greater_time = 1j * green_dual_greater_time \
                * screened_dual_lesser_time
            tmp_gw_2_retarded_time = (2 * np.pi) * np.array(
                [du.heaviside(t, 0) * (
                    tmp_gw_2_greater_time[i] - tmp_gw_2_lesser_time[i])
                    for i, t in enumerate(self.time)])
            n_dual = simps(green_dual_lesser.imag,
                           self.green_sys.freq)

            sigm_gw_channel_lesser_time = tmp_gw_1_lesser_time \
                + tmp_gw_2_lesser_time
            sigm_gw_channel_retarded_time = tmp_gw_1_retarded_time \
                + tmp_gw_2_retarded_time

            sigm_gw_channel_lesser_freq = dft.dft(
                sigm_gw_channel_lesser_time, self.time, self.green_sys.freq,
                sign=1)
            sigm_gw_channel_retarded_freq = dft.dft(
                sigm_gw_channel_retarded_time, self.time, self.green_sys.freq,
                sign=1)

            sigma_dual_tmp += fg.FrequencyGreen(
                self.green_sys.freq, retarded=sigm_gw_channel_retarded_freq,
                keldysh=sigm_gw_channel_lesser_freq, fermionic=True,
                keldysh_comp='lesser',
                orbitals=self.green_aux.orbitals)

            screened_dual_ret_w0 = simps(screened_dual_retarded_time,
                                         self.time)
            screened_dual_adv_w0 = simps(screened_dual_advanced_time,
                                         self.time)
            screened_dual_les_w0 = simps(screened_dual_lesser_time,
                                         self.time)
            self.sigma_hartree += n_dual * (screened_dual_adv_w0
                                            + screened_dual_ret_w0) \
                - 2. * screened_dual_les_w0
        if self.keldysh_comp == "lesser":
            self.sigma_dual = sigma_dual_tmp
        elif self.keldysh_comp == "keldysh":
            self.sigma_dual.retarded = sigma_dual_tmp.retarded
            self.sigma_dual.keldysh = sigma_dual_tmp.get_keldysh()
        # plt.plot(self.sigma_dual.retarded.imag)

    def compute_green_dual(self) -> None:
        """Calculate the fermionic dual Green's function.
        """
        self.green_dual = (self.green_bare_dual.inverse() -
                           self.sigma_dual).inverse()

    def compute_green_system(self) -> None:
        """Calculate the system Green's function.
        """
        # tmp = (self.green_aux * self.delta_hyb).inverse()
        self.green_sys = (self.delta_hyb.inverse() + (
            self.green_aux * self.delta_hyb).inverse()
            * self.green_dual * (self.delta_hyb * self.green_aux).inverse())

    def solve_selfconsistent_dual_gw(self, err_tol=1e-7, iter_max=100):
        self.compute_polarization_aux()
        self.compute_green_bare_dual()
        self.compute_bare_dual_screened_interaction()
        sigma_dual_tmp = fg.FrequencyGreen(self.green_aux.freq,
                                           keldysh_comp=self.keldysh_comp)
        for ii in range(iter_max):
            if ii == 0:
                self.compute_polarization_dual(self.green_bare_dual)
            else:
                self.compute_polarization_dual()
            self.compute_dual_screened_interaction()
            if ii == 0:
                self.compute_sigma_dual(self.green_bare_dual)
            else:
                self.compute_sigma_dual()
            self.compute_green_dual()

            self.err_iterations_gw.append(opt.cost_function(self.sigma_dual,
                                                            sigma_dual_tmp,
                                                            normalize=False))
            if ii == 1:
                plt.plot((self.sigma_dual - sigma_dual_tmp).retarded.imag)
            sigma_dual_tmp = fg.FrequencyGreen(
                self.green_aux.freq,
                retarded=self.sigma_dual.retarded,
                keldysh=self.sigma_dual.keldysh,
                keldysh_comp=self.keldysh_comp)

            err_iter_print = round(self.err_iterations_gw[-1], int(
                15 - np.log10(err_tol)))
            print(f'aux QME GW iter: {ii}\t|\tError:'
                  + f' {err_iter_print}')
            if self.err_iterations_gw[-1]  \
                    < err_tol:
                break
        self.compute_green_system()

    def save(self, fname: str, dir_: str, dataname: str,
             save_input_param: bool = True, save_aux_data: bool = False
             ) -> None:
        """Save the dual and auxiliary quantities to file.
        The auxilary Green's function and hybridization are passed to the
        class and therefore not saved.

        Parameters
        ----------
        fname : str
            File name to store data too.

        dir_ : str
            Group/Directory to save data too

        dataname : str
            Name under which to save T-DTRILEX data.

         save_input_param : str
            Save input parameters like green_aux, hyb_sys, etc, by default
            True.

        save_aux_data : bool, optional
            Save auxiliary objects calculated here like auxiliary polarization
            etc., by default False
        """
        hd5.add_attrs(fname, f"{dir_}/{dataname}", self.U_trilex)

        if save_input_param:
            freq = {'freq_min': self.hyb_aux.freq[0],
                    'freq_max': self.hyb_aux.freq[-1],
                    'N_freq': len(self.hyb_aux.freq)}
            time = {'time_min': self.time[0],
                    'time_max': self.time[-1],
                    'N_time': len(self.time)}
            hd5.add_attrs(fname, "/", {'freq': freq, 'time': time})

            self.green_aux.save(fname, '/auxiliary_sys',
                                'green_aux', savefreq=False)
            self.hyb_aux.save(fname, '/auxiliary_sys', 'hyb_aux',
                              savefreq=False)
            if ('hyb_dmft' not in hd5.get_directorys(fname, '/system') or
                    'hyb_leads' not in hd5.get_directorys(fname, '/system')):
                self.hyb_sys.save(fname, '/system', 'hyb_sys', savefreq=False)

        self.green_bare_dual.save(fname, f"{dir_}/{dataname}",
                                  "green_bare_dual_fermion",
                                  savefreq=False)
        self.green_dual.save(fname, f"{dir_}/{dataname}",
                             "green_dual", savefreq=False)
        self.sigma_dual.save(fname, f"{dir_}/{dataname}",
                             "sigma_dual", savefreq=False)

        # TODO: saving the bare propagators should be optional
        for channel in self.bare_dual_screened_interaction:
            self.bare_dual_screened_interaction[channel].save(
                fname, f"{dir_}/{dataname}/bare_dual_screened_interaction",
                f"{channel}", savefreq=False)

            if save_aux_data:
                self.polarization_aux[channel].save(
                    fname, f"{dir_}/{dataname}/polarization_aux", f"{channel}",
                    savefreq=False)

                self.susceptibility_aux[channel].save(
                    fname, f"{dir_}/{dataname}/susceptibility_aux",
                    f"{channel}", savefreq=False)

    def load(self, fname: str, dir_: str, dataname: str,
             load_input_param: bool = True, load_aux_data: bool = False
             ) -> None:
        """Load the dual and auxiliary quantities from file.
        The auxilary Green's function and hybridization are passed to the
        class and therefore not saved.

        Parameters
        ----------
        fname : str
            File name to store data too.

        dir_ : str
            Group/Directory to save data too.

        dataname : str
            Name under which to save T-DTRILEX data.

        load_input_param : bool, optional
            Load input parameters like green_aux, hyb_sys, etc, by default
            True.

        load_aux_data : bool, optional
            Load auxiliary objects calculated here like auxiliary polarization
            etc., by default False
        """
        self.U_trilex = hd5.read_attrs(fname, f"{dir_}/{dataname}")
        if self.correlators is None:
            spin_sector_max = 2
            aux_param = hd5.read_attrs(fname, '/auxiliary_sys')
            sys_param = hd5.read_attrs(fname, '/system')
            super_fermi_ops = sf_sub.SpinSectorDecomposition(
                nsite=aux_param['nsite'], spin_sector_max=spin_sector_max,
                spinless=sys_param['spinless'],
                tilde_conjugationrule_phase=sys_param['tilde_conjugation'])

            L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
            self.correlators = corr.Correlators(L, trilex=True)

        self.correlators.Lindbladian.load(fname, '/auxiliary_sys')
        self.correlators.update(T_mat=self.correlators.Lindbladian.T_mat,
                                U_mat=self.correlators.Lindbladian.U_mat,
                                Gamma1=self.correlators.Lindbladian.Gamma1,
                                Gamma2=self.correlators.Lindbladian.Gamma2)
        if load_input_param:
            grid = hd5.read_attrs(fname, '/')

            freq = np.linspace(
                grid['freq']['freq_min'], grid['freq']['freq_max'],
                grid['freq']['N_freq'])

            self.time = np.linspace(
                grid['time']['time_min'], grid['time']['time_max'],
                grid['time']['N_time'])

            self.green_aux = fg.FrequencyGreen(freq)
            self.green_aux.load(fname, '/auxiliary_sys',
                                'green_aux', readfreq=False)

            self.hyb_aux = fg.FrequencyGreen(freq)
            self.hyb_aux.load(fname, '/auxiliary_sys',
                              'hyb_aux', readfreq=False)

            if ('hyb_dmft' in hd5.get_directorys(fname, '/system') and
                    'hyb_leads' in hd5.get_directorys(fname, '/system')):
                hyb_dmft = fg.FrequencyGreen(freq)
                hyb_leads = fg.FrequencyGreen(freq)

                hyb_dmft.load(fname, '/system', 'hyb_dmft', readfreq=False)
                if 'hyb_leads' in hd5.get_directorys(fname, '/system'):
                    hyb_leads.load(fname, '/system',
                                   'hyb_leads', readfreq=False)

                self.hyb_sys = hyb_dmft + hyb_leads
            else:
                self.hyb_sys.load(fname, '/system', 'hyb_sys', readfreq=False)
            self.delta_hyb = self.hyb_aux - self.hyb_sys
            self.__init__(U_trilex=self.U_trilex, green_aux=self.green_aux,
                          hyb_sys=self.hyb_sys, hyb_aux=self.hyb_aux,
                          correlators=self.correlators)

        self.green_bare_dual.load(fname, f"{dir_}/{dataname}",
                                  "green_bare_dual",
                                  readfreq=False)
        self.green_dual.load(fname, f"{dir_}/{dataname}",
                             "green_dual_fermion", readfreq=False)
        self.sigma_dual.load(fname, f"{dir_}/{dataname}",
                             "sigma_dual", readfreq=False)

        # TODO: saving the bare propagators should be optional
        for channel in self.bare_dual_screened_interaction:
            self.bare_dual_screened_interaction[channel].load(
                fname, f"{dir_}/{dataname}/bare_dual_screened_interaction",
                f"{channel}", readfreq=False)

            # self.green_dual_boson[channel].load(
            #     fname, f"{dir_}/{dataname}/green_dual_boson",
            #     f"{channel}", readfreq=False)

            if load_aux_data:
                self.polarization_aux[channel].load(
                    fname, f"{dir_}/{dataname}/polarization_aux", f"{channel}",
                    readfreq=False)

                self.susceptibility_aux[channel].load(
                    fname, f"{dir_}/{dataname}/susceptibility_aux",
                    f"{channel}", readfreq=False)

        temp = hd5.read_dict_data(
            fname, f"{dir_}/{dataname}", 'three_point_vertex')
        for key in self.three_point_vertex:
            self.three_point_vertex[key] = temp[f"{key}"]

        # temp = hd5.read_dict_data(
        #     fname, f"{dir_}/{dataname}", 'four_point_vertex')
        # for key in self.four_point_vertex.keys():
        #     self.four_point_vertex[key] = temp[f"{key}"]


if __name__ == '__main__':
    aux_qme_dmft = aux_dmft.AuxiliaryMaserEquationDMFT(
        fname='auxiliaryDMFT.h5')
    U = aux_qme_dmft.parameters['system']['U']
    U_trilex = {'ch': U / 2, 'x': -U / 2, 'y': -U / 2, 'z': -U / 2}
    time_param = {'time_min': -10, 'time_max': 10, 'N_time': 1001}
    aux_dual_GW = AuxiliaryDualGW(time_param=time_param, U_trilex=U_trilex,
                                  green_aux=aux_qme_dmft.green_aux,
                                  hyb_sys=(aux_qme_dmft.hyb_leads
                                           + aux_qme_dmft.hyb_dmft),
                                  hyb_aux=aux_qme_dmft.hyb_aux,
                                  correlators=aux_qme_dmft.correlators,
                                  keldysh_comp=aux_qme_dmft.parameters[
                                      'system']['keldysh_comp'])

if __name__ == '__main__':
    aux_dual_GW.solve_selfconsistent_dual_gw(iter_max=10, err_tol=1e-10)

# %%
if __name__ == '__main__':
    plt.plot(aux_dual_GW.green_sys.retarded.imag)
    plt.plot(aux_dual_GW.green_sys.retarded.imag[::-1])
# %%
