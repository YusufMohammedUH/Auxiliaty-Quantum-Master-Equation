# %%
from typing import Dict, Union, Tuple, List
import numpy as np
import src.super_fermionic_space.model_lindbladian as lind
import src.greens_function.frequency_greens_function as fg
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.auxiliary_mapping.optimization_auxiliary_hybridization as opt
import src.greens_function.correlation_functions as corr
import src.impurity_solver.auxiliary_gw as gw
import src.dmft.dmft_base as dmft_base
import src.util.hdf5_util as hd5
import matplotlib.pyplot as plt


class AuxiliaryMaserEquationDualTrilexDMFT(dmft_base.DMFTBase):
    """Auxiliary master equation dynamical mean field theory solver class.
    The clase is derived from the DMFTBase class and implements the
    auxiliary master equation DMFT imputiy_solver.

    Parameters
    ----------
    dmft_base : dmft_base.DMFTBase
        DMFT base class object.

    parameters : Dict
        Dictionary containing the parameters of the DMFT solver.

    correlators : corr.Correlators
        Correlators object containing the Lindbladian and subroutines to
        calculate correlation (Green's) functions.

    hyb_leads : Union[fg.FrequencyGreen, None], optional
        Hybridization function of the leads, by default None.

    keldysh_comp : str, optional
        Keldysh component to be used, by default "keldysh".

    Attributes
    ----------
    correlators : corr.Correlators
        Correlators object, with which correlation functions can be calculated.

    T_mat : np.ndarray (dim, dim)
        Hopping matrix of the auxiliary system.

    U_mat : np.ndarray (dim)
        Onsite interaction matrix of the auxiliary system.

    green_aux : fg.FrequencyGreen
        Green's function of the auxiliary system.

    hyb_aux : fg.FrequencyGreen
        Hybridization function of the auxiliary system.

    keldysh_comp : str, optional
        Keldysh component to be used, by default "keldysh".
    """

    def __init__(self, parameters: Union[Dict, None] = None,
                 correlators: Union[corr.Correlators, None] = None,
                 hyb_leads: Union[fg.FrequencyGreen, None] = None,
                 fname: Union[str, None] = None) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        try:
            self.U_trilex = parameters['system']['U_trilex']
        except KeyError:
            U = parameters['system']['U']
            self.U_trilex = {'ch': U / 2, 'x': -
                             U / 2, 'y': -U / 2, 'z': -U / 2}

        self.T_mat = None
        self.U_mat = None
        self.green_aux = None
        self.hyb_aux = None
        self.aux_dual_gw = None
        self.aux_hyb = None
        dmft_base.DMFTBase.__init__(self, parameters=parameters,
                                    hyb_leads=hyb_leads, fname=fname)
# -------------------------- setup correletor class ------------------------- #
        if fname is None:
            self.correlators = correlators
        else:
            self.load(fname, read_parameters=True, load_parent=False)

        if self.parameters['aux_sys']['nsite']\
                - 2 * self.parameters['aux_sys']['Nb'] != 1:
            raise ValueError("Only single orbital included for now.")

        if self.correlators is None:
            raise ValueError("Correlators object is None.")

        if 'target_sites' not in self.parameters['aux_sys']:
            self.parameters['aux_sys']['target_sites'] = \
                self.correlators.Lindbladian.super_fermi_ops.target_sites
        self.set_local_matrix()

    def set_local_matrix(self, T_mat: Tuple[np.ndarray, None] = None,
                         U_mat: Tuple[np.ndarray, None] = None,
                         hubberdU: bool = True) -> None:
        """Set the hopping and interaction matrix of the auxiliary system.

        Parameters
        ----------
        T_mat : Tuple[np.ndarray, None], optional
            Hopping matrix of the auxiliary system, by default None.

        U_mat : Tuple[np.ndarray, None], optional
            Onsite interaction matrix of the auxiliary system, by default None.

        hubberdU : bool, optional
            Since the Hubbard interaction is site local the interaction matrix
            can be described by the main diagonal only(1d array),
            by default True
        """
        if T_mat is None:
            self.T_mat = np.zeros((self.parameters['aux_sys']['nsite'],
                                   self.parameters['aux_sys']['nsite']))
        else:
            self.T_mat = T_mat
        if U_mat is None:
            if hubberdU:
                self.U_mat = np.zeros(self.parameters['aux_sys']['nsite'])
                self.U_mat[self.parameters['aux_sys']['target_sites']] = \
                    self.parameters['system']['U']
            else:
                self.U_mat = np.zeros((self.parameters['aux_sys']['nsite'],
                                       self.parameters['aux_sys']['nsite'],
                                       self.parameters['aux_sys']['nsite'],
                                       self.parameters['aux_sys']['nsite']))
        else:
            self.U_mat = U_mat

        self.aux_hyb = opt.AuxiliaryHybridization(
            self.parameters['aux_sys']['Nb'],
            x_start=[0., 0.1, 0.5, -0.1, 0.2],
            U_mat=self.U_mat)

    def impurity_solver(self, optimization_options: Dict = {
            "disp": False, "maxiter": 500, 'ftol': 1e-5},
            x_start: List = [0., 0.1, 0.5, -0.1, 0.2]) -> None:
        # Optimization for determining the auxiliary hybridization function
        hyb_tot = None
        if not ((self.hyb_leads.keldysh_comp == self.hyb_dmft.keldysh_comp)
                and (self.hyb_leads.keldysh_comp == self.keldysh_comp)
                and (self.green_sys.keldysh_comp == self.keldysh_comp)):
            raise ValueError("ERROR: keldysh_comp of the leads and DMFT"
                             + " have to be the same as the"
                             + f" self.keldysh_comp: {self.keldysh_comp}")

        hyb_tot = self.hyb_leads + self.hyb_dmft

        self.hyb_aux = self.aux_hyb.update(
            hyb=hyb_tot, options=optimization_options)

        # #### Calculate the auxiliary single particle Green's function ###
        self.T_mat = self.aux_hyb.aux_sys.E

        self.correlators.update(T_mat=self.T_mat, U_mat=self.U_mat,
                                Gamma1=self.aux_hyb.aux_sys.Gamma1,
                                Gamma2=self.aux_hyb.aux_sys.Gamma2)

        self.green_aux = self.correlators.get_single_particle_green_physical(
            self.green_sys.freq, keldysh_comp=self.keldysh_comp)
        # plt.plot(self.green_aux.freq, self.green_aux.retarded.imag)
        self.aux_dual_gw = gw.AuxiliaryDualGW(
            time_param=self.parameters['time'],
            U_trilex=self.U_trilex, green_aux=self.green_aux,
            hyb_sys=hyb_tot, hyb_aux=self.hyb_aux,
            correlators=self.correlators, keldysh_comp=self.keldysh_comp)

        self.aux_dual_gw.solve(
            self.parameters['dual_GW']['err_tol'],
            self.parameters['dual_GW']['iter_max'])
        # ################### Extract the self-energy  ####################
        self.green_sys = self.aux_dual_gw.green_sys.copy()
        self.self_energy_int = self.green_sys.get_self_enerqy() \
            - hyb_tot
        return optimization_options, x_start

    def solve(self, optimization_options: Dict = {
            "disp": False, "maxiter": 500, 'ftol': 1e-5},
            x_start: List = [0., 0.1, 0.5, -0.1, 0.2]):
        self.__solve__((optimization_options, x_start))

    def save(self, fname: str) -> None:
        super().save(fname)
        self.green_aux.save(fname, '/auxiliary_sys',
                            'green_aux', savefreq=False)
        self.hyb_aux.save(fname, '/auxiliary_sys', 'hyb_aux', savefreq=False)
        hd5.add_attrs(fname, '/auxiliary_sys', self.parameters['aux_sys'])
        self.correlators.Lindbladian.save(fname, '/auxiliary_sys')

    def load(self, fname: str, read_parameters: bool = True,
             load_parent: bool = True) -> None:
        if load_parent:
            super().load(fname=fname, read_parameters=read_parameters)
        if read_parameters:
            self.parameters['aux_sys'] = hd5.read_attrs(
                fname, '/auxiliary_sys')

        freq = np.linspace(self.parameters['freq']['freq_min'],
                           self.parameters['freq']['freq_max'],
                           self.parameters['freq']['N_freq'])

        if self.green_aux is None:
            self.green_aux = fg.FrequencyGreen(
                freq, keldysh_comp=self.keldysh_comp)

        if self.hyb_aux is None:
            self.hyb_aux = fg.FrequencyGreen(
                freq, keldysh_comp=self.keldysh_comp)

        self.green_aux.load(fname, '/auxiliary_sys',
                            'green_aux', readfreq=False)
        self.hyb_aux.load(fname, '/auxiliary_sys', 'hyb_aux', readfreq=False)

        tilde = self.parameters['aux_sys']['tilde_conjugationrule_phase']
        super_fermi_ops = sf_sub.SpinSectorDecomposition(
            self.parameters['aux_sys']['nsite'],
            self.parameters['aux_sys']['spin_sector_max'],
            spinless=self.parameters['aux_sys']['spinless'],
            tilde_conjugationrule_phase=tilde)
        L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
        self.correlators = corr.Correlators(L, trilex=True)

        self.correlators.Lindbladian.load(fname, '/auxiliary_sys')
        self.correlators.update(T_mat=self.correlators.Lindbladian.T_mat,
                                U_mat=self.correlators.Lindbladian.U_mat,
                                Gamma1=self.correlators.Lindbladian.Gamma1,
                                Gamma2=self.correlators.Lindbladian.Gamma2)


if __name__ == "__main__":
    #  Frequency grid
    N_freq = 1000
    freq_max = 5

    N_time = 1000
    time_max = 20

    selfconsist_param = {'max_iter': 50, 'err_tol': 1e-6, 'mixing': 0.2}

    e0 = 0
    mu = 0
    beta = 10
    D = 25.1
    gamma = 0.1

    leads_param = {'e0': e0, 'mu': [mu], 'beta': beta, 'D': D, 'gamma': gamma}

    spinless = False
    spin_sector_max = 2
    tilde_conjugationrule_phase = True

    U = 3.0
    v = 1.0
    keldysh_comp = 'keldysh'
    sys_param = {'keldysh_comp': keldysh_comp, 'v': v, 'U': U, 'orbitals': 1}

    # Parameters of the auxiliary system
    Nb = 1
    nsite = 2 * Nb + 1
    aux_param = {'Nb': Nb, 'nsite': nsite, 'spinless': spinless,
                 'tilde_conjugationrule_phase': tilde_conjugationrule_phase,
                 'spin_sector_max': spin_sector_max}

    dual_GW = {'err_tol': 1e-7, 'iter_max': 50}
    params = {'time': {"time_min": -time_max, "time_max": time_max,
                       'N_time': N_time},
              'freq': {"freq_min": -freq_max, "freq_max": freq_max,
                       'N_freq': N_freq},
              'selfconsistency': selfconsist_param, 'leads': leads_param,
              'dual_GW': dual_GW,
              'aux_sys': aux_param, 'system': sys_param}

    # ##################### Initializing Lindblad class #######################
    super_fermi_ops = sf_sub.SpinSectorDecomposition(
        nsite, spin_sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)

    L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
    corr_cls = corr.Correlators(L)

    auxiliaryDMFT = AuxiliaryMaserEquationDualGWDMFT(
        parameters=params, correlators=corr_cls)
    auxiliaryDMFT.hyb_leads = auxiliaryDMFT.get_bath()
    auxiliaryDMFT.solve()

# %%
