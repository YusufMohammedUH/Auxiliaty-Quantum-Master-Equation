"""Here the base code for the construction of correlation functions
    will be written.
    """
# %%
from typing import Tuple, Union, List
import numpy as np
import src.greens_function.contour_util as con_util
import src.super_fermionic_space.model_lindbladian as lind
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.solvers.ed_solver as ed_sol
import src.auxiliary_mapping.auxiliary_system_parameter as aux
import src.greens_function.frequency_greens_function as fg

# XXX: works only for a single impurity site of interest
# XXX: works only for spin 1/2 fermions

# TODO: Restructure:
#       - Correlators should be a interface class
#       - a Solver should be supplied to the class:
#               i)  getting the steady state density of state
#               ii) get correlators: i)   ED
#                                    ii)  Lanczos/Arnoldi
#                                    iii) Tensornetwork/MPS/DMRG
#
#           -> should the solvers be children of the correlator class or
#              should the correlator have an attribute
#               !! Best use attributes -> can be used independently of rest!!
#
#       - should should it do:
#           - have subspace Lindbladian
#           - have the subspace creator and annihilator
#           - set everything before solving expectation values
#               - ordering and prefactors of expectation values
#               - sectors
#       - The rest should be moved in a ED solver class which


class Correlators:
    """Class for calculating correlators on the keldysh contour.

    """

    def __init__(self, Lindbladian: lind.Lindbladian, solver=ed_sol.EDSolver,
                 spin_components: Union[dict, None] = None,
                 correlators: Union[List, None] = None,
                 trilex: bool = False) -> None:
        """Container for calcutation of correlation function.
        It is assumed, that the steady state density operator is unique
        and located in the spin sector (0,0).

        It has to be considered in what limits this is valid.
        If valid the Lindbladian can be decomposed to spin sector and the time
        evolution can be extracted through exact diagonalization of this
        sectors.

        A spin sector is given by the difference between "normal" and "tilde"
        space particle number difference for spin up, first element of the
        tuple and spin down, second element of the tuple.

        Parameters
        ----------
        Lindbladian : src.model_lindbladian.Lindbladian
            Container class for the Lindbladian. A class attribute is the
            liouville space in which it is defined.

        """
        self.trilex = trilex
        self.Lindbladian = Lindbladian
        self.nsite = self.Lindbladian.super_fermi_ops.fock_ops.nsite
        self.operator_sectors = {"cdag": {"up": (1, 0), 'do': (0, 1)},
                                 'c': {'up': (-1, 0), 'do': (0, -1)}}

        if correlators is None:
            correlators = [2 * i for i in range(
                1, self.Lindbladian.super_fermi_ops.spin_sector_max + 1)]

        if self.trilex:
            correlators.append(3)

        self.set_contour_symmetries(correlators, trilex)

        self.operators_default_order = {2: ("c", "cdag"),
                                        3: ('c', 'cdag', 'rho'),
                                        4: ('c', 'c', 'cdag', 'cdag')}
        self.set_spin_components(spin_components)

        self.correlators = None
        # self.precalc_correlator = {n: {} for n in correlators}

        self.set_correlator_keys(correlators)

        self.solver = ed_sol.EDSolver(self.Lindbladian,
                                      (self.Lindbladian.super_fermi_ops
                                       ).spin_sectors)
        self.solver.prepare(self.correlators)

    def set_spin_components(self, spin_components: Union[dict, None] = None
                            ) -> None:
        """Set the desired spin components to calculate.

        Parameters
        ----------
        spin_components : dict, optional
            Dictionary containing the desired spin components of n-point
            correlators, by default None
        """
        if spin_components is None:
            self.spin_components = {2: [("up", "up")],
                                    3: [('up', 'up', ('up', 'up')),
                                        ('up', 'up', ('do', 'do')),
                                        ('up', 'do', ('up', 'do')),
                                        ('do', 'up', ('do', 'up'))],
                                    4: [('up', 'up', 'up', 'up'),
                                        ('up', 'do', 'up', 'do')]}
        else:
            self.spin_components = spin_components

    def set_correlator_keys(self, correlators: Union[dict, None] = None
                            ) -> None:
        """Set the dictionary keys of self.correlators. The attribute
        self.correlators is used to stor the n-point correlators.


        Parameters
        ----------
        correlators : _type_, optional
            _description_, by default None
        """
        if correlators is not None:
            self.correlators = {n: {} for n in correlators}
        else:
            self.correlators = {n: {} for n in self.correlators.keys()}

        for n in self.correlators.keys():

            for s in self.spin_components[n]:
                self.correlators[n][s] = {}

                for comp in con_util.get_branch_combinations(n):
                    self.correlators[n][s][comp] = {}

    def get_rho_steady_state(self) -> None:
        """Calculate the steady state density of states.
        It has to be unique.
        Wrapper to solver member function.
        """
        self.solver.get_rho_steady_state(self.Lindbladian)

    def update(self, T_mat: np.ndarray = None, U_mat: np.ndarray = None,
               Gamma1: np.ndarray = None, Gamma2: np.ndarray = None) -> None:
        """Update the Lindbladian and eigen decomposition.

        Parameters
        ----------
        T_mat : np.ndarray, optional
            Hopping matrix, by default None

        U_mat : np.ndarray, optional
            Interaction matrix, by default None

        Gamma1 : np.ndarray, optional
            Dissipative matrix, by default None

        Gamma2 : np.ndarray, optional
            Dissipative matrix, by default None
        """
        self.Lindbladian.update(T_mat=T_mat, U_mat=U_mat, Gamma1=Gamma1,
                                Gamma2=Gamma2, sign=1)
        self.get_rho_steady_state()
        if not self.Lindbladian.super_fermi_ops.tilde_conjugationrule_phase:
            self.Lindbladian.update(T_mat=T_mat, U_mat=U_mat, Gamma1=Gamma1,
                                    Gamma2=Gamma2, sign=-1)

        self.solver.update(self.Lindbladian)
        self.set_correlator_keys()

    def append_spin_sector_keys(self, operator_list: List[Tuple[str]]
                                ) -> List[Tuple]:
        """Append a tuple of spin sectors to the operators in the operator list
        e.g. [('c','up'),('cdag','do')] -> [('c','up',((-1,1),(0,1))),
        ('cdag','do',((0,1),(0,0)))]

        Warning: The steady state density matrix is always in the (0,0) sector,
        therefore the list right most and left most sector have to be (0,0).

        Parameters
        ----------
        operaotr_list : List[Tuple[str]]
            List of tuples of operators, e.g. [('c','up'),('cdag','do')]

        Returns
        -------
        List[Tuple]
            List of operators and spin sectors, e.g.
            [('c','up',((-1,1),(0,1))),('cdag','do',((0,1),(0,0)))]
        """
        if self.Lindbladian.super_fermi_ops.fock_ops.spinless:
            sectors = [0]
            for op in operator_list[::-1]:
                sectors.append(
                    self.Lindbladian.super_fermi_ops.get_left_sector(
                        sectors[-1], op))
        else:
            sectors = [(0, 0)]
            for op in operator_list[::-1]:
                sectors.append(
                    self.Lindbladian.super_fermi_ops.get_left_sector(
                        sectors[-1], op[0], op[1]))
        sectors.reverse()
        assert sectors[0] == sectors[-1]
        sector_keys = [x for x in zip(sectors[:-1], sectors[1:])]
        return tuple([(*op[0], op[1]) for op in zip(operator_list,
                                                    sector_keys)])

    def get_single_particle_green(self, component, freq, sites=None,
                                  spin=('up', 'up')):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        component : _type_
            _description_
        freq : _type_
            _description_
        sites : _type_, optional
            _description_, by default None
        spin : tuple, optional
            _description_, by default ('up', 'up')

        Returns
        -------
        _type_
            _description_
        """
        if sites is None:
            site = int(
                (self.Lindbladian.super_fermi_ops.fock_ops.nsite - 1) / 2)
            sites = (site, site)
        permutation_sign = None
        operators = None

        if component == (1, 0):
            permutation_sign = 1 + 0j
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('cdag', spin[1]),
                                         ('cdag_tilde', spin[0])),
                                        (('c', spin[0]),
                                         ('cdag', spin[1]))]]

        elif component == (0, 1):
            permutation_sign = -1 + 0j
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('cdag', spin[1]),
                                         ('c', spin[0])),
                                        (('c', spin[0]),
                                         ('c_tilde', spin[1]))]]

        return self.solver.get_correlator(Lindbladian=self.Lindbladian,
                                          freq=freq, component=component,
                                          sites=sites, operator_keys=operators,
                                          permutation_sign=permutation_sign)

    def get_three_point_vertex_components(
            self, component, freq, sites=None, spin=('up', 'up', ('up', 'up')),
            permutation_sign=(-1 + 0j, 1 + 0j, 1 + 0j), prefactor=-1 + 0j):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        component : _type_
            _description_
        freq : _type_
            _description_
        sites : _type_, optional
            _description_, by default None
        spin : tuple, optional
            _description_, by default ('up', 'up', ('up', 'up'))
        permutation_sign : tuple, optional
            _description_, by default (-1 + 0j, 1 + 0j, 1 + 0j)
        prefactor : _type_, optional
            _description_, by default -1+0j
        """
        operators = None
        if sites is None:
            site = int(
                (self.Lindbladian.super_fermi_ops.fock_ops.nsite - 1) / 2)
            sites = (site, site, site)

        # precalculating the expectation value
        if component == (0, 0, 0):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('c', spin[0]),
                                         ('cdag', spin[1]),
                                         ('rho', spin[2])),

                                        (('c', spin[0]),
                                         ('rho', spin[2]),
                                         ('cdag', spin[1])),

                                        (('cdag', spin[1]),
                                        ('c', spin[0]),
                                         ('rho', spin[2])),

                                        (('cdag', spin[1]),
                                         ('rho', spin[2]),
                                         ('c', spin[0])),

                                        (('rho', spin[2]),
                                         ('c', spin[0]),
                                         ('cdag', spin[1])),

                                        (('rho', spin[2]),
                                         ('cdag', spin[1]),
                                         ('c', spin[0]))]]

        elif component == (1, 0, 0):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('c', spin[0]),
                                         ('cdag', spin[1]),
                                         ('rho', spin[2])),

                                        (('cdag', spin[1]),
                                         ('rho', spin[2]),
                                         ('cdag_tilde', spin[0])),

                                        (('c', spin[0]),
                                         ('rho', spin[2]),
                                         ('cdag', spin[1])),

                                        (('rho', spin[2]),
                                         ('cdag', spin[1]),
                                         ('cdag_tilde', spin[0]))
                                        ]]

        elif component == (0, 1, 0):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('cdag', spin[1]),
                                         ('c', spin[0]),
                                         ('rho', spin[2])),

                                        (('c', spin[0]),
                                         ('rho', spin[2]),
                                         ('c_tilde', spin[1])),

                                        (('cdag', spin[1]),
                                         ('rho', spin[2]),
                                         ('c', spin[0])),

                                        (('rho', spin[2]),
                                         ('c', spin[0]),
                                         ('c_tilde', spin[1]))]]

        elif component == (0, 0, 1):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('rho', spin[2]),
                                         ('c', spin[0]),
                                         ('cdag', spin[1])),

                                        (('c', spin[0]),
                                        ('cdag', spin[1]),
                                         ('rho_tilde', spin[2][::-1])),

                                        (('rho', spin[2]),
                                         ('cdag', spin[1]),
                                         ('c', spin[0])),

                                        (('cdag', spin[1]),
                                        ('c', spin[0]),
                                         ('rho_tilde', spin[2][::-1]))
                                        ]]

        elif component == (1, 1, 0):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('c', spin[0]),
                                         ('cdag', spin[1]),
                                         ('rho', spin[2])),

                                        (('cdag', spin[1]),
                                         ('rho', spin[2]),
                                         ('cdag_tilde', spin[0])),

                                        (('cdag', spin[1]),
                                         ('rho', spin[2]),
                                         ('cdag_tilde', spin[0])),

                                        (('cdag', spin[1]),
                                         ('c', spin[0]),
                                         ('rho', spin[2])),

                                        (('c', spin[0]),
                                         ('rho', spin[2]),
                                         ('c_tilde', spin[1])),

                                        (('c', spin[0]),
                                         ('rho', spin[2]),
                                         ('c_tilde', spin[1]))]]

        elif component == (1, 0, 1):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('c', spin[0]),
                                         ('rho', spin[2]),
                                         ('cdag', spin[1])),

                                        (('rho', spin[2]),
                                         ('cdag', spin[1]),
                                         ('cdag_tilde', spin[0])),

                                        (('rho', spin[2]),
                                         ('cdag', spin[1]),
                                         ('cdag_tilde', spin[0])),

                                        (('rho', spin[2]),
                                         ('c', spin[0]),
                                         ('cdag', spin[1])),

                                        (('c', spin[0]),
                                        ('cdag', spin[1]),
                                         ('rho_tilde', spin[2][::-1])),

                                        (('c', spin[0]),
                                        ('cdag', spin[1]),
                                         ('rho_tilde', spin[2][::-1]))]]

        elif component == (0, 1, 1):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('cdag', spin[1]),
                                         ('rho', spin[2]),
                                         ('c', spin[0])),

                                        (('rho', spin[2]),
                                         ('c', spin[0]),
                                         ('c_tilde', spin[1])),

                                        (('rho', spin[2]),
                                         ('c', spin[0]),
                                         ('c_tilde', spin[1])),

                                        (('rho', spin[2]),
                                         ('cdag', spin[1]),
                                         ('c', spin[0])),

                                        (('cdag', spin[1]),
                                        ('c', spin[0]),
                                         ('rho_tilde', spin[2][::-1])),

                                        (('cdag', spin[1]),
                                        ('c', spin[0]),
                                         ('rho_tilde', spin[2][::-1]))
                                        ]]

        elif component == (1, 1, 1):
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in [(('cdag', spin[1]),
                                         ('rho', spin[2]),
                                         ('cdag_tilde', spin[0])),

                                        (('rho', spin[2]),
                                         ('cdag', spin[1]),
                                         ('cdag_tilde', spin[0])),

                                        (('c', spin[0]),
                                         ('rho', spin[2]),
                                         ('c_tilde', spin[1])),

                                        (('rho', spin[2]),
                                         ('c', spin[0]),
                                         ('c_tilde', spin[1])),

                                        (('c', spin[0]),
                                        ('cdag', spin[1]),
                                         ('rho_tilde', spin[2][::-1])),

                                        (('cdag', spin[1]),
                                        ('c', spin[0]),
                                         ('rho_tilde', spin[2][::-1]))
                                        ]]

        return self.solver.get_correlator(Lindbladian=self.Lindbladian,
                                          freq=freq, component=component,
                                          sites=sites, operator_keys=operators,
                                          permutation_sign=permutation_sign)

    def get_three_point_vertex(self, freq: np.ndarray,
                               spin: Tuple = ('up', 'up', ('up', 'up')),
                               permutation_sign: Tuple = (-1 +
                                                          0j, 1 + 0j, 1 + 0j),
                               prefactor: complex = -1 + 0j,
                               return_: bool = False):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        freq : _type_
            _description_
        spin : tuple, optional
            _description_, by default ('up', 'up', ('up', 'up'))
        permutation_sign : tuple, optional
            _description_, by default (-1 + 0j, 1 + 0j, 1 + 0j)
        prefactor : _type_, optional
            _description_, by default -1+0j
        return_ : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if not return_:
            for component in self.correlators[3][spin]:
                self.correlators[3][spin][component] = \
                    self.get_three_point_vertex_components(
                        component, freq, permutation_sign=permutation_sign,
                        prefactor=prefactor)
        else:
            three_point_vertex = np.zeros(
                (freq.shape[0], freq.shape[0], 2, 2, 2),
                dtype=np.complex128)
            for i, j, k in self.correlators[3][
                    ("up", "up", ("up", "up"))]:
                three_point_vertex[:, :, i, j, k] = \
                    self.get_three_point_vertex_components(
                        (i, j, k), freq, spin,
                    permutation_sign=permutation_sign, prefactor=prefactor)

            return three_point_vertex

    def set_contour_symmetries(self, n_correlators: dict,
                               trilex: bool = False) -> None:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        n_correlators : dict
            _description_
        trilex : bool, optional
            _description_, by default False
        """
        self.contour_symmetries = {}
        n_corr = n_correlators.copy()

        for n in n_corr:
            contour = con_util.get_branch_combinations(n)
            t_max = con_util.t_max_possitions(n)
            all_combinations = []
            for cs in contour:
                for ts in t_max:
                    tmp = []
                    for c, t in zip(cs, ts):
                        tmp.append((c, t))
                    all_combinations.append(tuple(tmp))

            n_correlator_symmetrie = {}
            for comb in all_combinations:
                if comb not in n_correlator_symmetrie.keys():
                    t_max_idx = con_util.find_index_max_time(comb)

                    if comb[t_max_idx][0] == 1:
                        n_correlator_symmetrie[comb] = None
                        comb_symmetric = list(comb)
                        comb_symmetric[t_max_idx] = tuple(
                            [x if i != 0 else 0 for i, x in
                             enumerate(comb[t_max_idx])])
                        n_correlator_symmetrie[tuple(comb_symmetric)] = comb
            if n == 2:
                con_util.choose_components([((1, 1), (0, 0)),
                                            ((1, 0), (0, 1)),
                                            ((0, 1), (1, 0)),
                                            ((0, 0), (1, 1))],
                                           n_correlator_symmetrie)
            if n == 3 and trilex:
                self.contour_symmetries[3] = \
                    n_correlator_symmetrie.copy()
            else:
                self.contour_symmetries[n] = n_correlator_symmetrie.copy()


# %%
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # Set parameters
    ws = np.linspace(-10, 10, 200)

    Nb = 1
    nsite = 2 * Nb + 1
    U_imp = 1.0
    es = np.array([1])
    ts = np.array([0.5])
    Us = np.zeros(nsite)
    Us[Nb] = U_imp
    gamma = np.array([0.2 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])

    # Initializing Lindblad class
    spin_sector_max = 2
    spinless = False
    tilde_conjugationrule_phase = True
    super_fermi_ops = sf_sub.SpinSectorDecomposition(
        nsite, spin_sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)
    # plt.figure()
    L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
    # for U in [2.0]:

    # Initializing auxiliary system and E, Gamma1 and Gamma2 for a
    # particle-hole symmetric system
    sys = aux.AuxiliarySystem(Nb, ws)
    sys.set_ph_symmetric_aux(es, ts, gamma)

    green = fg.FrequencyGreen(sys.ws)
    green.set_green_from_auxiliary(sys)
    hyb_aux = green.get_self_enerqy()

    # Setting unitary part of Lindbladian
    T_mat = sys.E
    T_mat[Nb, Nb] = -Us[Nb] / 2.0
    # L.update(T_mat=T_mat, U_mat=Us, Gamma1=sys.Gamma1, Gamma2=sys.Gamma2)

    # Setting dissipative part of Lindbladian
    # L.set_dissipation(sys.Gamma1, sys.Gamma2)
    print("after setting dissipator")
    # Setting total Lindbladian
    # L.set_total_linbladian()

    # Setup a correlator object
    corr = Correlators(Lindbladian=L, trilex=True)
    corr.update(T_mat=T_mat, U_mat=Us, Gamma1=sys.Gamma1,
                Gamma2=sys.Gamma2)
    # # corr.update_model_parameter(sys.Gamma1, sys.Gamma2, T_mat, Us)
    # corr.set_rho_steady_state(set_lindblad=False)
    # corr.sectors_exact_decomposition(set_lindblad=False)

    # Calcolate Green's functions
    G_lesser_plus, G_lesser_minus = corr.get_single_particle_green(
        (0, 1), ws)
    G_greater_plus, G_greater_minus = corr.get_single_particle_green(
        (1, 0), ws)
    G_les = G_lesser_plus + G_lesser_minus
    G_gr = G_greater_plus + G_greater_minus
    G_R = G_greater_plus - G_lesser_plus
    G_K = G_greater_plus + G_greater_minus + G_lesser_plus + G_lesser_minus

    # green2 = fg.FrequencyGreen(sys.ws, retarded=G_R, keldysh=G_K)
    # sigma = green2.get_self_enerqy() - hyb_aux

    # # Visualize results

    # # plt.plot(sys.ws, green.retarded.imag)
    # plt.figure()
    # plt.plot(sys.ws, G_R.imag)
    # plt.ylabel(r"$A(\omega)$")
    # plt.xlabel(r"$\omega$")
    # plt.show()
    # plt.figure()
    # plt.plot(sys.ws, G_K.imag)
    # plt.ylabel(r"$A(\omega)$")
    # plt.xlabel(r"$\omega$")
    # plt.show()
    # plt.legend([r"$U = 0$",r"$U = 0.5$",r"$U = 1$",r"$U = 1.5$",
    #                 r"$U = 2$"])
    # plt.show()

    # plt.figure()
    # plt.plot(sys.ws, green.keldysh.imag)
    # plt.plot(sys.ws, G_K.imag)
    # plt.xlabel(r"$\omega$")
    # plt.legend([r"$ImG^K_{aux,0}(\omega)$",
    #             r"$ImG^K_{aux}(\omega)$"])
    # plt.show()

    # plt.figure()
    # plt.plot(sys.ws, hyb_aux.retarded.imag)
    # plt.plot(sys.ws, sigma.retarded.imag)
    # plt.xlabel(r"$\omega$")
    # plt.legend([r"$Im\Delta^R_{aux}(\omega)$",
    #             r"$Im\Sigma^R_{aux}(\omega)$"])
    # plt.show()

# %%
