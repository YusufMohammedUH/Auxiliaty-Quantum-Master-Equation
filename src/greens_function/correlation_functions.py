"""Here the base code for the construction of correlation functions
    will be written.
    """
# %%
from typing import Dict, Tuple, Union, List, Iterable
import numpy as np
from itertools import product
import src.greens_function.contour_util as con_util
import src.super_fermionic_space.model_lindbladian as lind
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.liouville_solvers.ed_solver as ed_sol
import src.auxiliary_mapping.auxiliary_system_parameter as aux
import src.greens_function.frequency_greens_function as fg
import src.greens_function.correlator_components as comp

# XXX: works only for a single impurity site of interest
# XXX: works only for spin 1/2 fermions


class Correlators:
    """Class for calculating correlators on the keldysh contour.
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

    solver : _type_, optional
            _description_, by default ed_sol.EDSolver

    spin_components : Union[dict, None], optional
        _description_, by default None

    correlators : Union[List, None], optional
        _description_, by default None

    trilex : bool, optional
        _description_, by default False

    Attributes
    ----------
    trilex : bool
        Flag to indicate if the trilex correlators are to be calculated.

    Lindbladian : src.model_lindbladian.Lindbladian
        Container class for the Lindbladian.

    nsite : int
        Number of sites in the auxiliary system.

    operators_default_order : dict
        Dictionary containing the default order of the 2, 3 and 4 point
        correlators.

    correlators : dict
        Dictionary containing the correlators to be calculated.

    solver : Object
        Solver for calculating n-point correlators, e.g. of class EDSolver.

    spin_components : dict
        Dictionary containing the desired spin components of n-point
        correlators. These can be reduced due to symmetry, e.g. paramagnetism.
    """

    def __init__(self, Lindbladian: lind.Lindbladian, solver=ed_sol.EDSolver,
                 spin_components: Union[dict, None] = None,
                 correlators: Union[List, None] = None,
                 trilex: bool = False) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.trilex = trilex
        self.Lindbladian = Lindbladian
        self.nsite = self.Lindbladian.super_fermi_ops.fock_ops.nsite

        if correlators is None:
            correlators = [2 * i for i in range(
                1, self.Lindbladian.super_fermi_ops.spin_sector_max + 1)]

        if self.trilex:
            correlators.append(3)

        self.set_contour_symmetries(correlators, trilex)

        self.operators_default_order = {2: ("c", "cdag"),
                                        3: ('c', 'cdag', 'n_channel'),
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
            self.spin_components = {2: [("up", "up"), ('do', 'do')],

                                    3: [  # ch channel can be only nonzero for
                                        ('up', 'up', 'ch'),
                                        ('do', 'do', 'ch'),
                                        # x channel can be only nonzero for
                                        ('up', 'do', 'x'),
                                        ('do', 'up', 'x'),
                                        # y channel can be only nonzero for
                                        ('up', 'do', 'y'),
                                        ('do', 'up', 'y'),
                                        # should be zero in p-h symmetrie
                                        ('up', 'up', 'z'),
                                        ('do', 'do', 'z')],

                                    4: [('up', 'up', 'up', 'up'),
                                        ('up', 'up', 'do', 'do'),
                                        ('up', 'do', 'up', 'do'),

                                        ('do', 'up', 'do', 'up'),
                                        ('do', 'do', 'up', 'up'),
                                        ('do', 'do', 'do', 'do')]}
        else:
            self.spin_components = spin_components

    def set_correlator_keys(self, correlators: Union[dict, None] = None
                            ) -> None:
        """Set the dictionary keys of self.correlators. The attribute
        self.correlators is used to stor the n-point correlators.


        Parameters
        ----------
        correlators : Union[dict, None], optional
            Dictionary containing the components of the correlators, by default
            None.
        """
        if correlators is not None:
            self.correlators = {n: {} for n in correlators}
        else:
            self.correlators = {n: {} for n in self.correlators.keys()}

        for n in self.correlators.keys():

            for s in self.spin_components[n]:
                self.correlators[n][s] = {}

                for comp_ in con_util.get_branch_combinations(n):
                    self.correlators[n][s][comp_] = {}

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
        ('cdag','do',((0,1),(0,0)))]. In case of more then one configuration of
        sectors all possible sectors are added as a list of operator lists,
        e.g. [(n_channel,'x'),(n_channel,'y')]->
        [[(n_channel,'x',(1,-1)),(n_channel,'y',(-1,1))],
        [(n_channel,'x',(-1,1)),(n_channel,'y',(1,-1))]]

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
            sectors = [[(0, 0)]]
            for op in operator_list[::-1]:
                if 'x' == op[1] or 'y' == op[1]:
                    sector = self.Lindbladian.super_fermi_ops.operator_sectors[
                        op[0]][op[1]]
                else:
                    sector = [
                        self.Lindbladian.super_fermi_ops.operator_sectors[
                            op[0]][op[1]]]

                sectors.append(sector)
            sectors_list = list(product(*sectors))
            sector_added_list = []
            for sectors in sectors_list:
                tmp = (0, 0)
                tmp_list = []
                for sector in sectors:
                    tmp = sf_sub.add_spin_sectors(tmp, sector)
                    tmp_list.append(tmp)
                if tmp == (0, 0):
                    sector_added_list.append(tmp_list[::-1])
            sectors_keys_list = [[x for x in zip(sectors[:-1], sectors[1:])]
                                 for sectors in sector_added_list]
        assert len(sectors_keys_list) >= 1, 'sectors_keys_list is empty'

        return [tuple([(*op[0], op[1]) for op in zip(
            operator_list, sector_keys)]) for sector_keys in sectors_keys_list]

    def get_single_particle_green(self, component: Tuple[int, int],
                                  freq: np.ndarray,
                                  sites: Union[None, Tuple[int, int]] = None,
                                  spin: Tuple[str, str] = ('up', 'up')
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the single particle green's function for the desired contour
        component 'component' for given spin, site and frequency grid.

        Parameters
        ----------
        component : Tuple[int, int]
            Contour component, e.g. (0,0) or (1,0). 1 for th backward and 0 for
            the forward branch.

        freq : np.ndarray
            Frequency grid

        sites :  Union[None, Tuple[int, int]], optional
            Tuple of sites, by default None

        spin : Tuple[str,str], optional
            Tuple of spins, by default ('up', 'up')

        Returns
        -------
        out: Tuple[np.ndarray, np.ndarray]
            Tuple of two single particle green's functions
        """
        if sites is None:
            site = int(
                (self.Lindbladian.super_fermi_ops.fock_ops.nsite - 1) / 2)
            sites = (site, site)
        prefactor = -1j
        permutation_sign = None
        operators = None
        operator_components = comp.get_two_point_operator_list(spin=spin)
        if component == (0, 0):
            permutation_sign = (1. + 0.j, -1. + 0.j)

        elif component == (0, 1):
            permutation_sign = (-1. + 0.j, -1. + 0.j)

        elif component == (1, 0):
            permutation_sign = (1. + 0.j, 1. + 0.j)

        elif component == (1, 1):
            permutation_sign = (-1. + 0.j, 1. + 0.j)
        operators = [self.append_spin_sector_keys(op_key)[0]
                     for op_key in operator_components[component]]

        return self.solver.get_correlator(Lindbladian=self.Lindbladian,
                                          freq=freq, component=component,
                                          sites=sites, operator_keys=operators,
                                          permutation_sign=permutation_sign,
                                          prefactor=prefactor)

    def get_single_particle_green_physical(self, freq: np.ndarray,
                                           sites: Union[
                                               None, Tuple[int, int]] = None,
                                           spin: Tuple[str, str] = ('up', 'up')
                                           ) -> fg.FrequencyGreen:
        """Return the single particle green's function on the physical contour
        e.g. returns a fg.FrequencyGreen object.

        Parameters
        ----------
        freq : np.ndarray
            Frequency grid
        sites : Union[ None, Tuple[int, int]], optional
            Target sites, by default None
        spin : Tuple[str, str], optional
            Spin indices of the Green's function, by default ('up', 'up')

        Returns
        -------
        out: Tuple[np.ndarray, np.ndarray]
            Green's function on the physical contour, retarded and keldysh
            component
        """
        green_greater_plus, green_greater_minus = \
            self.get_single_particle_green((1, 0), freq, sites, spin)
        green_lesser_plus, green_lesser_minus = \
            self.get_single_particle_green((0, 1), freq, sites, spin)

        green_aux_R = green_greater_plus - green_lesser_plus
        green_aux_K = green_greater_plus + green_greater_minus \
            + green_lesser_plus + green_lesser_minus

        return fg.FrequencyGreen(
            freq, retarded=green_aux_R, keldysh=green_aux_K)

    def get_susceptibility(self, freq: np.ndarray, component: Tuple[int, int],
                           channels: Tuple[str, str],
                           sites: Union[None, Iterable] = None,
                           prefactor: complex = -1 + 0j):
        """Calculate and return the susceptibility for the desired component,
        channels and sites.


        Parameters
        ----------
        freq : np.ndarray
            Frequency grid

        component : Tuple[int, int]
            Contour component, e.g. (0,0) or (1,0). 1 for th backward and 0 for
            the forward branch.

        channels : Tuple[str, str]
            Charge 'ch' or spin channels 'x','y' or 'z'

        sites : Union[None, Iterable], optional
            Tuple of site indices, by default None

        prefactor : complex, optional
            prefactor of the susceptibility, by default -1+0j

        Returns
        -------
        out: tuple[np.ndarray, np.ndarray]
            Tuple of the positive and negative susceptibility components
            (FT(Kai(t < 0))(\omega), FT(Kai(t > 0))(\omega) ).
        """
        if sites is None:
            site = int(
                (self.Lindbladian.super_fermi_ops.fock_ops.nsite - 1) / 2)
            sites = (site, site)
        permutation_sign = (1. + 0.j, 1. + 0.j)
        operator_components = comp.get_susceptibility(channels=channels)

        if 'ch' in channels or 'z' in channels:
            operators = [[self.append_spin_sector_keys(op_key)[0]
                          for op_key in operator_components[component]]]
        else:
            operators = [self.append_spin_sector_keys(op_key)
                         for op_key in operator_components[component]]

        kai_plus = np.zeros(len(freq), dtype=np.complex128)
        kai_minus = np.zeros(len(freq), dtype=np.complex128)
        for ops in operators:
            kai_tmp_plus, kai_tmp_minus = self.solver.get_correlator(
                Lindbladian=self.Lindbladian, freq=freq, component=component,
                sites=sites, operator_keys=ops,
                permutation_sign=permutation_sign, prefactor=prefactor)

            kai_plus += kai_tmp_plus
            kai_minus += kai_tmp_minus

        return kai_plus, kai_minus

    def get_susceptibility_physical(self, freq: np.ndarray,
                                    channels: Tuple[str, str],
                                    sites: Union[None, Iterable] = None,
                                    prefactor: complex = - 1j) -> fg.FrequencyGreen:
        """Return the susceptibility on the physical contour e.g. returns a
        fg.FrequencyGreen object.

        Parameters
        ----------
        freq : np.ndarray
            Frequency grid
        channels : Tuple[str, str]
            Charge ('ch','ch') or spin channels ('x','x'),('x','y'), etc.
        sites : Union[None, Iterable], optional
            target sites, by default None
        prefactor : complex, optional
            prefactor of the susceptibility corresponding to (-i)^2
            for two particle green's function, by default -1+0j

        Returns
        -------
        out: fg.FrequencyGreen
            Susceptibility on the physical contour
        """
        chi_greater_plus, chi_greater_minus = \
            self.get_susceptibility(freq=freq, component=(1, 0),
                                    channels=channels,
                                    sites=sites, prefactor=prefactor)
        chi_lesser_plus, chi_lesser_minus = \
            self.get_susceptibility(freq=freq, component=(0, 1),
                                    channels=channels,
                                    sites=sites, prefactor=prefactor)

        chi_aux_R = chi_greater_plus - chi_lesser_plus
        chi_aux_K = chi_greater_plus + chi_greater_minus + chi_lesser_plus \
            + chi_lesser_minus

        return fg.FrequencyGreen(
            freq, retarded=chi_aux_R, keldysh=chi_aux_K)

    def get_three_point_vertex_components(
            self, component: Tuple[int, int, int], freq: np.ndarray,
            sites: Union[None, List, Tuple] = None,
            spin: Tuple[str, str, str] = ('up', 'up', 'ch'),
            permutation_sign: Tuple[int, int, int] = (-1 + 0j, 1 + 0j, 1 + 0j),
            prefactor: complex = -1 + 0j) -> np.ndarray:
        """Return the single particle green's function for the desired contour
        component 'component' for given spin, site and frequency grid.
        Parameters
        ----------
        component : Tuple[int, int, int]
            Contour component, e.g. (0,0,0) or (0, 1,0). 1 for th backward
            and 0 for the forward branch.

        freq : np.ndarray
            Frequency grid

        sites : Union[None, np.ndarray], optional
            Tuple of sites, by default None

        spin : Tuple[str, str, str], optional
            Spins of the creation and annihilation operators and
            channel of the spin and density operators, by default
            ('up', 'up', 'ch')

        permutation_sign : Tuple[int, int, int], optional
            Prefactor picked up by permutation, by default (-1 + 0j, 1 + 0j,
            1 + 0j)

        prefactor : complex, optional
            Prefactor corresponds to the sign of the two particle green's
            function and the chosen default order, by default -1+0j

        Returns
        -------
        out: np.ndarray (dim, dim)
            return the three point vertex of desired component, spin and site
        """
        operators = None
        if sites is None:
            site = int(
                (self.Lindbladian.super_fermi_ops.fock_ops.nsite - 1) / 2)
            sites = (site, site, site)
        operator_components = comp.get_three_point_operator_list(spin=spin)
        # precalculating the expectation value

        operators = [[self.append_spin_sector_keys(op_key)[0]
                      for op_key in operator_components[component]]]

        vertex = np.zeros((len(freq), len(freq)), dtype=np.complex128)
        for ops in operators:
            vertex_tmp = self.solver.get_correlator(
                Lindbladian=self.Lindbladian, freq=freq, component=component,
                sites=sites, operator_keys=ops,
                permutation_sign=permutation_sign, prefactor=prefactor)

            vertex += vertex_tmp

        return vertex

    def get_three_point_vertex(self, freq: np.ndarray,
                               spin: Tuple = ('up', 'up', 'ch'),
                               sites: Union[Tuple, None] = None,
                               permutation_sign: Tuple = (-1 +
                                                          0j, 1 + 0j, 1 + 0j),
                               prefactor: complex = -1 + 0j,
                               return_: bool = False) -> Union[None, Dict]:
        """Return the single particle green's function for the desired contour
        component 'component' for given spin, site and frequency grid.
        Parameters
        ----------
        component : Tuple[int, int, int]
            Contour component, e.g. (0,0,0) or (0, 1,0). 1 for th backward
            and 0 for the forward branch.

        freq : np.ndarray
            Frequency grid

        spin : Tuple[str, str, str], optional
            Spins of the creation and annihilation operators and
            channel of the spin and density operators, by default
            ('up', 'up', 'ch')

        sites : Union[None, np.ndarray], optional
            Tuple of sites, by default None

        permutation_sign : Tuple[int, int, int], optional
            Prefactor picked up by permutation, by default (-1 + 0j, 1 + 0j,
            1 + 0j)

        prefactor : complex, optional
            Prefactor corresponds to the sign of the two particle green's
            function and the chosen default order, by default -1+0j

        return_ : bool, optional
            If true the dictionary containing the three point vertex is
            returned, by default False

        Returns
        -------
        out: Union[None,Dict]
            return the three point vertex of desired spin and site and all
            components
        """

        if not return_:
            for component in self.correlators[3][spin]:
                self.correlators[3][spin][component] = \
                    self.get_three_point_vertex_components(
                        component=component, freq=freq, sites=sites, spin=spin,
                        permutation_sign=permutation_sign,
                        prefactor=prefactor)
        else:
            three_point_vertex = np.zeros(
                (freq.shape[0], freq.shape[0], 2, 2, 2),
                dtype=np.complex128)
            for i, j, k in self.correlators[3][
                    ("up", "up", 'ch')]:
                three_point_vertex[:, :, i, j, k] = \
                    self.get_three_point_vertex_components(
                        component=(i, j, k), freq=freq, sites=sites, spin=spin,
                    permutation_sign=permutation_sign, prefactor=prefactor)

            return three_point_vertex

    def set_contour_symmetries(self, n_correlators: Dict,
                               trilex: bool = False) -> None:
        """Calculate the contour symmetries and save them as dictionary
        self.contour_symmetries can be used to reduce computation

        Parameters
        ----------
        n_correlators : dict
            dictionary containing the correlators.

        trilex : bool, optional
            Calculate the symmetries of the 3 point vertex if set to True, by
            default False
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
    import matplotlib.pyplot as plt
    # Set parameters
    ws = np.linspace(-10, 10, 201)
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
    L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)

    # Initializing auxiliary system and E, Gamma1 and Gamma2 for a
    # particle-hole symmetric system
    sys = aux.AuxiliarySystem(Nb, ws)
    sys.set_ph_symmetric_aux(es, ts, gamma)

    G_aux_U0 = fg.FrequencyGreen(sys.ws)
    G_aux_U0.set_green_from_auxiliary(sys)
    hyb_aux = G_aux_U0.get_self_enerqy()

    # Setting unitary part of Lindbladian
    T_mat = sys.E
    T_mat[Nb, Nb] = -Us[Nb] / 2.0

    # Setup a correlator object
    corr = Correlators(Lindbladian=L, trilex=True)
    corr.update(T_mat=T_mat, U_mat=Us, Gamma1=sys.Gamma1,
                Gamma2=sys.Gamma2)

    # Calculate Green's functions
    G_lesser_plus, G_lesser_minus = corr.get_single_particle_green(
        (0, 1), ws)
    G_greater_plus, G_greater_minus = corr.get_single_particle_green(
        (1, 0), ws)

    G_R = G_greater_plus - G_lesser_plus
    G_K = G_greater_plus + G_greater_minus + G_lesser_plus + G_lesser_minus

    G_aux_full = fg.FrequencyGreen(sys.ws, retarded=G_R, keldysh=G_K)
    sigma = G_aux_full.get_self_enerqy() - hyb_aux

    # Visualize results

    plt.figure()
    plt.plot(sys.ws, G_aux_U0.retarded.imag)
    plt.plot(sys.ws, G_aux_full.retarded.imag)
    plt.ylabel(r"$A(\omega)$")
    plt.xlabel(r"$\omega$")
    plt.legend([r"$Im G^R_{aux0}(\omega)$",
                r"$Im G^R_{aux}(\omega)$"])
    plt.show()

    plt.figure()
    plt.plot(sys.ws, G_aux_U0.keldysh.imag)
    plt.plot(sys.ws, G_aux_full.keldysh.imag)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$ImG^K_{aux,0}(\omega)$",
                r"$ImG^K_{aux}(\omega)$"])
    plt.show()

    plt.figure()
    plt.plot(sys.ws, hyb_aux.retarded.imag)
    plt.plot(sys.ws, sigma.retarded.imag)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$Im\Delta^R_{aux}(\omega)$",
                r"$Im\Sigma^R_{aux}(\omega)$"])
    plt.show()

# %%
