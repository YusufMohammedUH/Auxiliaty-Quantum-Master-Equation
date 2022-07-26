# %%
from typing import Dict, Union, Callable, Tuple
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import src.super_fermionic_space.model_lindbladian as lind
import src.greens_function.frequency_greens_function as fg
import src.greens_function.dos_util as du
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.auxiliary_mapping.optimization_auxiliary_hybridization as opt
import src.greens_function.correlation_functions as corr
import src.hdf5_util as h5

# XXX: The method how the system greens function is calculated from the
#   system hybridization using the auxiliary method should be a
#   class or function passed to the class.
#       - allows for method agnostic usage of the class, e.g.
#         direct auxiliary mapping, auxiliary Dual Fermion,
#         auxiliary Dual TRILEX, etc.


class AuxiliaryMaserEquationDMFT:
    """Class for the DMFT solver using the auxiliary master equation method.

    _extended_summary_

    Parameters
    ----------
    parameters : Dict
        Contains all the parameters needed for the calculation.

    hyb_leads : Union[fg.FrequencyGreen, None], optional
        Lead hybridization function, by default None

    Attributes
    ----------
    parameters : Dict
        Contains all the parameters needed for the calculation.

    hyb_leads : Union[fg.FrequencyGreen, None]
        Lead hybridization function.

    hyb_dmft : fg.FrequencyGreen
        DMFT hybridization function.

    hyb_sys : fg.FrequencyGreen
        Total system hybridization(hyb_dmft+hyb_leads) function.

    green_sys : fg.FrequencyGreen
        Green's function of the system.

    green_aux : fg.FrequencyGreen
        Green's function of the auxiliary system.

    self_energy_int : fg.FrequencyGreen
        Self-energy of the system.

    T_mat : np.ndarray
        Hopping matrix of the auxiliary system.

    U_mat : np.ndarray
        Onsite interaction matrix of the auxiliary system.

    err_iterations : List
        List of the errors at each iteration.

    Raises
    ------
    ValueError
        Only single orbital included for now. Therefor must be
        nsite = 2*Nb + 1.
    """

    def __init__(self, parameters: Dict, hyb_leads:
                 Union[fg.FrequencyGreen, None] = None) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.parameters = parameters

        if parameters['aux_sys']['nsite']\
                - 2 * parameters['aux_sys']['Nb'] != 1:
            raise ValueError("Only single orbital included for now.")

        if hyb_leads is None:
            freq = np.linspace(parameters['freq']['freq_min'],
                               parameters['freq']['freq_max'],
                               parameters['freq']['N_freq'])
            freq.flags.writeable = False
            self.hyb_leads = fg.FrequencyGreen(freq)
            self.hyb_dmft = fg.FrequencyGreen(freq)
            self.green_sys = fg.FrequencyGreen(freq)
            self.green_aux = fg.FrequencyGreen(freq)
            self.self_energy_int = fg.FrequencyGreen(freq)
        else:
            self.hyb_leads = hyb_leads
            self.hyb_dmft = fg.FrequencyGreen(self.hyb_leads.freq)
            self.green_sys = fg.FrequencyGreen(self.hyb_leads.freq)
            self.green_aux = fg.FrequencyGreen(self.hyb_leads.freq)
            self.self_energy_int = fg.FrequencyGreen(self.hyb_leads.freq)

        self.err_iterations = []
        self.T_mat = None
        self.U_mat = None
        self.hyb_sys = None
        self.hyb_aux = None
        self.self_energy_int = None

    def get_bath(self, param: Union[Dict, None] = None,
                 bath_hyb_function: Callable[
        [float, float, float, float, float],
            Tuple[np.ndarray, np.ndarray]] = du.flat_bath_retarded
    ) -> fg.FrequencyGreen:
        """Return a hybridization function for the passed arguments.

        Parameters
        ----------
        param : Union[Dict, None], optional
            Dictionary containing the parameters passed to the function
            bath_hyb_function, by default None.
            If None than the parameters specified in self.parameters['leads']
            are used.

        bath_hyb_function : Function, optional
            Function returning the keldysh and retarded components
            (tuple of two np.ndarray's of complex numbers), by default
            du.flat_bath_retarded

        Returns
        -------
        out:    fg.FrequencyGreen
            Hybridization of the system leads.
        """
        if param is None:
            param = self.parameters
        tmp = fg.FrequencyGreen(self.green_sys.freq)
        for mu in param['leads']['mu']:
            args = np.array([param['leads']['e0'], mu,
                            param['leads']['beta'], param['leads']['D'],
                            param['leads']['gamma']],
                            dtype=np.float64)
            tmp += du.set_hybridization(self.hyb_leads.freq, bath_hyb_function,
                                        args)
        return tmp

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

    def solve(self, correlators: corr.Correlators,
              U_mat: Union[np.ndarray, None] = None) -> None:
        """Solve the auxiliary dynamic mean-field theory for a given set of
        parameters and a supplied object of class Correlators.

        Parameters
        ----------
        correlators : corr.Correlators
            Object of class Correlators containing routines to calculate the
            single particle greens function and the Lindblad operator.

        U_mat : Union[np.ndarray, None], optional
            Interaction matrix, by default None
        """

        if 'target_sites' not in self.parameters['aux_sys']:
            self.parameters['aux_sys']['target_sites'] = \
                correlators.Lindbladian.super_fermi_ops.target_sites
        self.err_iterations = []
        self.set_local_matrix(U_mat=U_mat)
        # ##################### Optimization parameters #######################
        x_start = [0., 0.1, 0.5, -0.1, 0.2]
        optimization_options = {"disp": False, "maxiter": 500, 'ftol': 1e-5}

        green_tmp = fg.FrequencyGreen(self.hyb_leads.freq)
        # ##################### initial Green's function ######################

        args = np.array([self.parameters['leads']['e0'],
                         0,
                         self.parameters['leads']['beta'], 1.0, 1.0],
                        dtype=np.float64)
        self.green_sys = du.set_hybridization(
            self.green_sys.freq, du.lorenzian_bath_retarded, args)
        for ii in range(self.parameters['selfconsistency']['max_iter']):
            # ###### Calculate the DMFT hybridization for a Bethe lattice. ####
            self.hyb_dmft = fg.FrequencyGreen(
                self.hyb_leads.freq, self.green_sys.retarded *
                (self.parameters['system']['v']**2),
                self.green_sys.keldysh * (self.parameters['system']['v']**2))
            # Optimization for determining the auxiliary hybridization function
            self.hyb_sys = self.hyb_leads + self.hyb_dmft
            optimal_param = opt.optimization_ph_symmertry(
                self.parameters['aux_sys']['Nb'], hybridization=(
                    self.hyb_sys),
                x_start=x_start,
                options=optimization_options
            )

            x_start = np.copy(optimal_param.x)

            aux_sys = opt.get_aux(
                optimal_param.x, self.parameters['aux_sys']['Nb'],
                self.hyb_leads.freq)
            self.hyb_aux = fg.get_hyb_from_aux(aux_sys)
            # #### Calculate the auxiliary single particle Green's function ###
            self.T_mat = aux_sys.E
            self.T_mat[self.parameters['aux_sys']['Nb'],
                       self.parameters['aux_sys']['Nb']] -= \
                self.parameters['system']['U'] / 2.

            correlators.update(T_mat=self.T_mat, U_mat=self.U_mat,
                               Gamma1=aux_sys.Gamma1,
                               Gamma2=aux_sys.Gamma2)

            G_greater_plus, G_greater_minus = \
                correlators.get_single_particle_green(
                    (1, 0), self.hyb_leads.freq)
            G_lesser_plus, G_lesser_minus = \
                correlators.get_single_particle_green(
                    (0, 1), self.hyb_leads.freq)

            green_aux_R = G_greater_plus - G_lesser_plus
            green_aux_K = G_greater_plus + G_greater_minus + G_lesser_plus \
                + G_lesser_minus

            self.green_aux = fg.FrequencyGreen(
                aux_sys.ws, retarded=green_aux_R, keldysh=green_aux_K)
            # ################### Extract the self-energy  ####################
            self.self_energy_int = self.green_aux.get_self_enerqy() \
                - self.hyb_aux

            # ##### Calculate the system single particle Green's function #####
            green_tmp.dyson(aux_sys.ws, self.hyb_sys + self.self_energy_int)

            self.err_iterations.append(opt.cost_function(self.green_sys,
                                                         green_tmp,
                                                         normalize=False))

            self.green_sys = fg.FrequencyGreen(aux_sys.ws, retarded=(
                (1.0 - self.parameters['selfconsistency']['mixing'])
                * green_tmp.retarded
                + self.parameters['selfconsistency']['mixing']
                * self.green_sys.retarded),
                keldysh=((1.0 - self.parameters['selfconsistency']['mixing'])
                         * green_tmp.keldysh
                         + self.parameters['selfconsistency']['mixing']
                         * self.green_sys.keldysh))

            if self.err_iterations[-1]  \
                    < self.parameters['selfconsistency']['err_tol']:
                break
            spectral_weight = round((-1 / np.pi) * simps(
                self.green_sys.retarded.imag, self.green_sys.freq), 8)
            # XXX: occupation should be also printed. Therefore I should figure
            #   out how to get the occupation from the keldysh green
            #   function. Or directly work with the lesser Green's function
            #   instead.
            err_iter_print = round(self.err_iterations[-1], int(
                5 - np.log10(self.parameters['selfconsistency']['err_tol'])))
            print(
                f"Iteration No.: {ii}   |    Error: "
                + "{0:.7}   |   ".format(err_iter_print)
                + f"Spectral weight: {spectral_weight}")
        self.hyb_dmft = fg.FrequencyGreen(
            self.hyb_leads.freq, self.green_sys.retarded *
            (self.parameters['system']['v']**2),
            self.green_sys.keldysh * (self.parameters['system']['v']**2))
        self.hyb_sys = self.hyb_leads + self.hyb_sys

    def save(self, fname: str) -> None:
        """Save data to file

        Parameters
        ----------
        fname : str
            Name of HDF5 file.
        """
        h5.create_hdf5(fname)
        h5.add_attrs(fname, '/', self.parameters['freq'])
        h5.add_data(fname, '/', 'convergence', self.err_iterations)
        h5.add_attrs(fname, '/convergence', self.parameters['selfconsistency'])

        self.green_sys.save(fname, '/system', 'green_sys')
        self.hyb_dmft.save(fname, '/system', 'hyb_dmft')
        h5.add_attrs(fname, '/system', self.parameters['system'])

        if 'leads' in self.parameters['leads']:
            self.hyb_leads.save(fname, '/system', 'hyb_leads')
            h5.add_attrs(fname, '/system', self.parameters['leads'])

        self.green_aux.save(fname, '/auxiliary_sys', 'green_aux')
        self.hyb_aux.save(fname, '/auxiliary_sys', 'hyb_aux')
        h5.add_attrs(fname, '/auxiliary_sys', self.parameters['aux_sys'])


# %%
if __name__ == "__main__":
    #  Frequency grid
    N_freq = 400
    freq_max = 10

    selfconsist_param = {'max_iter': 50, 'err_tol': 1e-7, 'mixing': 0.2}

    e0 = 0
    mu = 0
    beta = 100
    D = 10
    gamma = 0.1

    leads_param = {'e0': e0, 'mu': [mu], 'beta': beta, 'D': D, 'gamma': gamma}

    spinless = False
    spin_sector_max = 1
    tilde_conjugationrule_phase = True

    U = 3.0
    v = 1.0
    sys_param = {'v': v, 'U': U, 'spinless': spinless,
                 'tilde_conjugation': tilde_conjugationrule_phase}

    # Parameters of the auxiliary system
    Nb = 1
    nsite = 2 * Nb + 1
    aux_param = {'Nb': Nb, 'nsite': nsite}

    params = {'freq': {"freq_min": -freq_max, "freq_max": freq_max,
                       'N_freq': N_freq},
              'selfconsistency': selfconsist_param, 'leads': leads_param,
              'aux_sys': aux_param, 'system': sys_param}

    # ##################### Initializing Lindblad class #######################
    super_fermi_ops = sf_sub.SpinSectorDecomposition(
        nsite, spin_sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)

    L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
    corr_cls = corr.Correlators(L)

    auxiliaryDMFT = AuxiliaryMaserEquationDMFT(params)
    auxiliaryDMFT.hyb_leads = auxiliaryDMFT.get_bath()
    auxiliaryDMFT.solve(corr_cls)
    # auxiliaryDMFT.save('auxiliaryDMFT.hdf5')
# %%