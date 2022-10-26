# %%
from typing import Dict, Union, Tuple, List
import numpy as np
import src.super_fermionic_space.model_lindbladian as lind
import src.greens_function.frequency_greens_function as fg
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.auxiliary_mapping.optimization_auxiliary_hybridization as opt
import src.greens_function.correlation_functions as corr
import src.greens_function.convert_keldysh_components as conv
import src.dmft.dmft as dmft
import src.util.hdf5_util as hd5


class AuxiliaryMaserEquationDMFT(dmft.DMFTBase):
    def __init__(self, parameters: Dict, correlators: corr.Correlators,
                 hyb_leads: Union[fg.FrequencyGreen, None] = None,
                 keldysh_comp: str = "keldysh") -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        super().__init__(parameters, hyb_leads, keldysh_comp)
        self.correlators = correlators
        self.T_mat = None
        self.U_mat = None

        if parameters['aux_sys']['nsite']\
                - 2 * parameters['aux_sys']['Nb'] != 1:
            raise ValueError("Only single orbital included for now.")

        if 'target_sites' not in self.parameters['aux_sys']:
            self.parameters['aux_sys']['target_sites'] = \
                self.correlators.Lindbladian.super_fermi_ops.target_sites

        self.green_aux = fg.FrequencyGreen(self.green_sys.freq)

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

    def impurity_solver(self, optimization_options: Dict = {
            "disp": False, "maxiter": 500, 'ftol': 1e-5},
            x_start: List = [0., 0.1, 0.5, -0.1, 0.2]) -> None:
        # Optimization for determining the auxiliary hybridization function
        if self.keldysh_comp == "lesser":
            hyb_tot = self.hyb_leads + self.hyb_dmft
            hyb_keldysh = conv.get_keldysh_from_lesser(hyb_tot)
            hyb_tot.keldysh = hyb_keldysh
        elif self.keldysh_comp == "keldysh":
            hyb_tot = self.hyb_leads + self.hyb_dmft

        optimal_param = opt.optimization_ph_symmertry(
            self.parameters['aux_sys']['Nb'], hybridization=hyb_tot,
            x_start=x_start,
            options=optimization_options
        )
        x_start = np.copy(optimal_param.x)

        aux_sys = opt.get_aux(
            optimal_param.x, self.parameters['aux_sys']['Nb'],
            self.green_sys.freq)
        self.hyb_aux = fg.get_hyb_from_aux(aux_sys)
        # #### Calculate the auxiliary single particle Green's function ###
        self.T_mat = aux_sys.E
        self.T_mat[self.parameters['aux_sys']['Nb'],
                   self.parameters['aux_sys']['Nb']] -= \
            self.parameters['system']['U'] / 2.

        self.correlators.update(T_mat=self.T_mat, U_mat=self.U_mat,
                                Gamma1=aux_sys.Gamma1,
                                Gamma2=aux_sys.Gamma2)

        self.green_aux = self.correlators.get_single_particle_green_physical(
            self.green_sys.freq)
        if self.keldysh_comp == "lesser":
            self.green_aux.keldysh = conv.get_lesser_from_keldysh(
                self.green_aux)
        # ################### Extract the self-energy  ####################
        self.self_energy_int = self.green_aux.get_self_enerqy() \
            - self.hyb_aux
        self.green_sys = fg.FrequencyGreen(freq=self.green_sys.freq)
        self.green_sys.dyson(self_energy=(
            self.hyb_dmft + self.hyb_leads + self.self_energy_int))

        return optimization_options, x_start

    def solve(self, optimization_options: Dict = {
            "disp": False, "maxiter": 500, 'ftol': 1e-5},
            x_start: List = [0., 0.1, 0.5, -0.1, 0.2]):
        self.__solve__((optimization_options, x_start))

    def save_child_data(self, fname: str) -> None:
        self.green_aux.save(fname, '/auxiliary_sys',
                            'green_aux', savefreq=False)
        self.hyb_aux.save(fname, '/auxiliary_sys', 'hyb_aux', savefreq=False)
        hd5.add_attrs(fname, '/auxiliary_sys', self.parameters['aux_sys'])
        self.correlators.Lindbladian.save(fname, '/auxiliary_sys')

    def load_child_data(self, fname: str, read_parameters: bool = False
                        ) -> None:
        self.green_aux.load(fname, '/auxiliary_sys',
                            'green_aux', readfreq=False)
        self.hyb_aux.load(fname, '/auxiliary_sys', 'hyb_aux', readfreq=False)
        self.correlators.Lindbladian.load(fname, '/auxiliary_sys')
        self.correlators.update(T_mat=self.correlators.Lindbladian.T_mat,
                                U_mat=self.correlators.Lindbladian.U_mat,
                                Gamma1=self.correlators.Lindbladian.Gamma1,
                                Gamma2=self.correlators.Lindbladian.Gamma2)

        if read_parameters:
            self.parameters['aux_sys'] = hd5.read_attrs(
                fname, '/auxiliary_sys')


if __name__ == "__main__":
    #  Frequency grid
    N_freq = 400
    freq_max = 10

    selfconsist_param = {'max_iter': 50, 'err_tol': 1e-6, 'mixing': 0.2}

    e0 = 0
    mu = 0
    beta = 10
    D = 10.1
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

    auxiliaryDMFT = AuxiliaryMaserEquationDMFT(params, correlators=corr_cls,
                                               keldysh_comp='lesser')
    auxiliaryDMFT.hyb_leads = auxiliaryDMFT.get_bath()
    auxiliaryDMFT.set_local_matrix()
    auxiliaryDMFT.solve()
    # auxiliaryDMFT.save('auxiliaryDMFT.h5')

# %%
