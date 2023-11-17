# %%
from typing import Dict, Optional
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import src.greens_function.frequency_greens_function as fg
import src.greens_function.correlation_functions as corr
import src.util.hdf5_util as hd5
import src.greens_function.dos_util as du
import src.util.fourier as dft
import src.impurity_solver.auxiliary_solver_base as aux_base
# import src.dmft.auxiliary_dmft as aux_dmft

import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.super_fermionic_space.model_lindbladian as lind


class AuxiliaryDualGW(aux_base.AuxiliaryDualSolverBase):
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
                 time_param: Optional[np.ndarray] = None,
                 U_trilex: Optional[Dict] = None,
                 green_aux: Optional[fg.FrequencyGreen] = None,
                 hyb_sys: Optional[fg.FrequencyGreen] = None,
                 hyb_aux: Optional[fg.FrequencyGreen] = None,
                 correlators: Optional[corr.Correlators] = None,
                 keldysh_comp: str = 'keldysh') -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        super().__init__(filename=filename, dir_=dir_, dataname=dataname,
                         load_input_param=load_input_param,
                         load_aux_data=load_aux_data,
                         U_trilex=U_trilex,
                         green_aux=green_aux,
                         hyb_sys=hyb_sys,
                         hyb_aux=hyb_aux,
                         correlators=correlators,
                         keldysh_comp=keldysh_comp)

        self.time = np.linspace(time_param['time_min'],
                                time_param['time_max'],
                                time_param['N_time'])

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
        green_retarded_time = 2 * np.pi * np.array(
            [du.heaviside(t, 0) * (
                green_greater_time[i] - green_lesser_time[i])
             for i, t in enumerate(self.time)])

        polarizability_lesser_time = 1j * green_lesser_time * \
            green_greater_time.conj()

        polarizability_retarded_time = 1j * green_retarded_time \
            * green_lesser_time.conj() \
            - 1j * green_lesser_time * green_retarded_time.conj()

        polarizability_tmp = fg.FrequencyGreen(
            freq=self.green_sys.freq,
            retarded=dft.dft(polarizability_retarded_time, self.time,
                             self.green_sys.freq, sign=1),
            keldysh=dft.dft(polarizability_lesser_time,
                            self.time, self.green_sys.freq, sign=1),
            fermionic=False,
            keldysh_comp='lesser',
            orbitals=self.green_sys.orbitals)

        if self.keldysh_comp == 'lesser':
            self.polarization_dual = polarizability_tmp
        elif self.keldysh_comp == 'keldysh':
            self.polarization_dual.retarded = polarizability_tmp.retarded
            self.polarization_dual.keldysh = polarizability_tmp.get_keldysh()
            self.polarization_dual.keldysh_comp = self.keldysh_comp

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
        # green_dual_retarded_time = 2 * np.pi * np.array(
        #     [du.heaviside(t, 0) * (
        #         green_dual_greater_time[i] - green_dual_lesser_time[i])
        #      for i, t in enumerate(self.time)])
        self.sigma_hartree = 0
        sigma_dual_tmp = fg.FrequencyGreen(
            freq=self.green_aux.freq,
            keldysh_comp='lesser',
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
            # according to note of Zhenya only the above diagram contrebutes to
            # the dual self-energy
            # tmp_gw_2_lesser_time = -1j * green_dual_lesser_time \
            #     * screened_dual_greater_time.conj()

            # tmp_gw_2_retarded_time = -1j * green_dual_retarded_time\
            #     * screened_dual_lesser_time + green_dual_lesser_time\
            #     * screened_dual_retarded_time

            n_dual = simps(green_dual_lesser.imag,
                           self.green_sys.freq)

            sigm_gw_channel_lesser_time = \
                tmp_gw_1_lesser_time \
                # + tmp_gw_2_lesser_time
            sigm_gw_channel_retarded_time = \
                tmp_gw_1_retarded_time \
                # + tmp_gw_2_retarded_time

            sigm_gw_channel_lesser_freq = dft.dft(
                sigm_gw_channel_lesser_time, self.time, self.green_sys.freq,
                sign=1)
            sigm_gw_channel_retarded_freq = dft.dft(
                sigm_gw_channel_retarded_time, self.time, self.green_sys.freq,
                sign=1)

            sigma_dual_tmp += fg.FrequencyGreen(
                freq=self.green_sys.freq,
                retarded=sigm_gw_channel_retarded_freq,
                keldysh=sigm_gw_channel_lesser_freq,
                fermionic=True,
                keldysh_comp='lesser',
                orbitals=self.green_aux.orbitals)

            screened_dual_ret_w0 = simps(screened_dual_retarded_time,
                                         self.time)
            screened_dual_adv_w0 = simps(screened_dual_advanced_time,
                                         self.time)
            # Also here the diagrams are double counted the above seem to be
            # the right one.
            # screened_dual_les_w0 = simps(screened_dual_lesser_time,
            #                              self.time)

            # self.sigma_hartree += n_dual * (screened_dual_adv_w0
            #                                 + screened_dual_ret_w0) \
            #     - 2. * screened_dual_les_w0
            self.sigma_hartree += n_dual * (screened_dual_adv_w0
                                            + screened_dual_ret_w0)

        if self.keldysh_comp == "lesser":
            self.sigma_dual = sigma_dual_tmp
        elif self.keldysh_comp == "keldysh":
            self.sigma_dual.retarded = sigma_dual_tmp.retarded
            self.sigma_dual.keldysh = sigma_dual_tmp.get_keldysh()

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

    def load(self, fname: str, dir_: str, dataname: str,
             load_input_param: bool = True, load_aux_data: bool = False
             ) -> None:
        self.__load__(fname=fname, dir_=dir_, dataname=dataname,
                      load_input_param=load_input_param,
                      load_aux_data=load_aux_data)

        if load_input_param:
            grid = hd5.read_attrs(fname, '/')

            self.time = np.linspace(
                grid['time']['time_min'], grid['time']['time_max'],
                grid['time']['N_time'])


if __name__ == '__main__':
    import src.greens_function.dos_util as dos
    import src.auxiliary_mapping.optimization_auxiliary_hybridization as opt
    # freq
    w_max = 10
    w_n = 1001
    freq = np.linspace(-w_max, w_max, w_n)
    # bath
    e0 = 0
    mu = 0
    beta = 100
    D = 1
    gamma = 0
    # auxiliary sites
    Nb = 1
    nsite = 2 * Nb + 1
    keldysh_comp = 'keldysh'
    U = 5.0
    Us = np.array([0., U, 0], dtype=np.complex128)
    hyb_sys = dos.set_hybridization(
        freq=freq,
        retarded_function=dos.semi_circular_bath_retarded,
        args=[e0, mu, beta, D, gamma], keldysh_comp=keldysh_comp)

    aux_hyb = opt.AuxiliaryHybridization(
        Nb=Nb,
        x_start=[0., 0.1, 0.5, -0.1, 0.2],
        U_mat=Us)

    hyb_aux = aux_hyb.update(hyb=hyb_sys)
    green_aux = fg.FrequencyGreen(freq=freq,
                                  keldysh_comp=keldysh_comp)
    green_aux.dyson(hyb_aux)

    U_charge = U / 2
    U_spin = -U / 2
    U_trilex = {'ch': U_charge, 'x': U_spin, 'y': U_spin, 'z': U_spin}
    time_param = {'time_min': -10, 'time_max': 10, 'N_time': 1001}
    # correlator
    spinless = False
    spin_sector_max = 2
    tilde_conjugationrule_phase = True

    super_fermi_ops = sf_sub.SpinSectorDecomposition(
        nsite, spin_sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)
    L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)

    # Setup a correlator object
    corr_cls = corr.Correlators(Lindbladian=L, trilex=True)
    corr_cls.update(T_mat=aux_hyb.aux_sys.E, U_mat=Us,
                    Gamma1=aux_hyb.aux_sys.Gamma1,
                    Gamma2=aux_hyb.aux_sys.Gamma2)

    aux_dual_GW = AuxiliaryDualGW(time_param=time_param, U_trilex=U_trilex,
                                  green_aux=green_aux,
                                  hyb_sys=hyb_sys,
                                  hyb_aux=hyb_aux,
                                  correlators=corr_cls,
                                  keldysh_comp=keldysh_comp)
if __name__ == '__main__':
    aux_dual_GW.solve(iter_max=10, err_tol=1e-10)
    # plt.plot(aux_dual_GW.green_sys.retarded.imag)
    # plt.plot(aux_dual_GW.green_sys.retarded.imag[::-1])
#  %%
if __name__ == '__main__':
    plt.plot(freq, aux_dual_GW.hyb_sys.keldysh.imag, label='original')
    plt.plot(freq, aux_dual_GW.green_aux.keldysh.imag, label='aqm')
    plt.plot(freq, aux_dual_GW.green_sys.keldysh.imag, label='aqm-gw')
    plt.legend()
    plt.show()

    plt.plot(freq, aux_dual_GW.hyb_sys.retarded.imag, label='original')
    plt.plot(freq, aux_dual_GW.green_aux.retarded.imag, label='aqm')
    # plt.plot(freq, aux_dual_GW.green_sys.retarded.imag, label='aqm-gw')
    plt.plot(freq, aux_dual_GW.green_sys.retarded.imag, label='aqm-gw')
    plt.legend()
    plt.show()

    plt.plot(freq, np.pi * aux_dual_GW.green_sys.keldysh.imag, label='aqm-gw')
    plt.plot(freq, aux_dual_GW.green_sys.retarded.imag, label='aqm-gw')
    plt.legend()
    plt.show()
# %%
if __name__ == '__main__':
    aux_qme_dmft = aux_dmft.AuxiliaryMaserEquationDMFT(
        fname='auxiliaryDMFT.h5')
    U = aux_qme_dmft.parameters['system']['U']
    U_trilex = {'ch': U / 2, 'x': -U / 2, 'y': -U / 2, 'z': -U / 2}
    time_param = {'time_min': -10, 'time_max': 10, 'N_time': 1000}
    aux_dual_GW2 = AuxiliaryDualGW(time_param=time_param, U_trilex=U_trilex,
                                   green_aux=aux_qme_dmft.green_aux,
                                   hyb_sys=(aux_qme_dmft.hyb_leads
                                            + aux_qme_dmft.hyb_dmft),
                                   hyb_aux=aux_qme_dmft.hyb_aux,
                                   correlators=aux_qme_dmft.correlators,
                                   keldysh_comp=aux_qme_dmft.parameters[
                                       'system']['keldysh_comp'])

# if __name__ == '__main__':
#     aux_dual_GW2.solve(iter_max=10, err_tol=1e-10)

# if __name__ == '__main__':
#     plt.plot(freq, aux_qme_dmft.green_sys.keldysh.imag, label='original')
#     plt.plot(freq, aux_dual_GW2.green_aux.keldysh.imag, label='aqm')
#     plt.plot(freq, aux_dual_GW2.green_sys.keldysh.imag, label='aqm-gw')
#     plt.legend()
#     plt.show()

#     plt.plot(freq, aux_qme_dmft.green_sys.retarded.imag, label='original')
#     plt.plot(freq, aux_dual_GW2.green_aux.retarded.imag, label='aqm')
#     plt.plot(freq, aux_dual_GW2.green_sys.retarded.imag, label='aqm-gw')
#     plt.legend()
#     plt.show()
    # %%
