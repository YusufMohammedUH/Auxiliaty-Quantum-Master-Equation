"""Here the base code for the construction of correlation functions
    will be written.
    """
# %%
import itertools
import numpy as np
from numba import njit, prange
import src.super_fermionic_space.model_lindbladian as lind
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.solvers.exact_decomposition as ed_lind
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
    def __init__(self, Lindbladian, spin_components=None,
                 correlators=None, trilex=False):
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

        self.set_contour_symmetries(correlators, trilex)

        if self.trilex:
            correlators.append('trilex')

        self.operators_default_order = {2: ("c", "cdag"),
                                        3: ('c', 'cdag', 'rho'),
                                        4: ('c', 'c', 'cdag', 'cdag')}
        self.set_spin_components(spin_components)

        self.precalc_correlator = {n: {} for n in correlators}

        self.set_correlator_keys(correlators)

    def set_spin_components(self, spin_components=None):
        if spin_components is None:
            self.spin_indices = {2: [("up", "up")],
                                 3: [('up', 'up', ('up', 'up')),
                                 ('up', 'up', ('do', 'do')),
                                     ('up', 'do', ('up', 'do')),
                                     ('do', 'up', ('do', 'up'))],
                                 4: [('up', 'up', 'up', 'up'),
                                     ('up', 'do', 'up', 'do')]}
        else:
            self.spin_indices = spin_components

    def set_correlator_keys(self, correlators=None):
        if correlators is not None:
            self.correlators = {n: {} for n in correlators}
        else:
            self.correlators = {n: {} for n in self.correlators.keys()}
        self.precalc_correlator_keys = {}

        for n in self.correlators.keys():
            if n == 'trilex':
                m = 3
            else:
                m = n
            self.precalc_correlator_keys[n] = {}
            for s in self.spin_indices[m]:
                self.correlators[n][s] = {}
                self.precalc_correlator_keys[n][s] = {}

                for comp in get_branch_combinations(m):
                    self.correlators[n][s][comp] = {}

    def get_rho_steady_state(self):
        """Calculate the steady state density of state in the spin sector
        (0,0).

        Raises
        ------
        ValueError
            If there are more then one steady-steate density of states
        """
        if self.Lindbladian.super_fermi_ops.fock_ops.spinless:
            sector0 = 0
        else:
            sector0 = (0, 0)

        L_00 = self.Lindbladian.super_fermi_ops.get_subspace_object(
            self.Lindbladian.L_tot, sector0, sector0)

        vals, _, vec_r = ed_lind.exact_spectral_decomposition(
            L_00.todense())

        mask = np.isclose(vals, np.zeros(vals.shape))
        vals_close_zero = vals[mask]
        n_steady_state = vals_close_zero.shape[0]
        if n_steady_state == 0:
            raise ValueError("No steady state density of state.")
        if n_steady_state > 1:
            vals_close_zero_sorted = sorted(
                vals_close_zero, key=(lambda x: np.abs(x)))
            vals_renormalized = [e / np.abs(vals_close_zero_sorted[0])
                                 for e in vals_close_zero_sorted]
            mask2 = vals_close_zero == vals_close_zero_sorted[0]
            if np.abs(vals_renormalized[1]) < 100:
                print("eigen values")
                raise ValueError(
                    "There are more than one stready states values")
            self.rho_stready_state = vec_r[mask][mask2][0]
        else:
            self.rho_stready_state = vec_r[mask][0]

        self.left_vacuum = \
            self.Lindbladian.super_fermi_ops.get_subspace_object(
                (self.Lindbladian.super_fermi_ops.left_vacuum
                 ).transpose().conjugate(),
                sector_right=sector0).todense()

        self.rho_stready_state /= self.left_vacuum * self.rho_stready_state

        # self.check_sector(self.rho_stready_state)

    def sectors_exact_decomposition(self):
        """Exactly decompose the Lindbladian within the relevant spin
        sectors. The eigenvectors and eigenvalues are saved as object
        attributes.

        Parameters
        ----------
        set_lindblad : bool, optional
            If True the Lindbladian is updated with a sign of -1, by default
            True. This is done to ensure that the Lindbladian describes the
            time propagation of a single femionic operator, with a single
            fermionic operator coupling the Markovian bath to the system.
        """

        self.vals_sector = {}
        self.vec_l_sector = {}
        self.vec_r_sector = {}

        for sector in self.Lindbladian.super_fermi_ops.spin_sectors:
            if not self.Lindbladian.super_fermi_ops.fock_ops.spinless:
                sector = tuple(sector)
            L_sector = self.Lindbladian.super_fermi_ops.get_subspace_object(
                self.Lindbladian.L_tot, sector, sector)
            self.vals_sector[sector], self.vec_l_sector[sector],\
                self.vec_r_sector[sector] = \
                ed_lind.exact_spectral_decomposition(L_sector.todense())

    def reset_correlator_data(self):
        self.precalc_correlator = {n: {} for n in self.correlators.keys()}
        self.set_correlator_keys()

    def update(self, T_mat=None, U_mat=None, Gamma1=None, Gamma2=None):
        self.Lindbladian.update(T_mat=T_mat, U_mat=U_mat, Gamma1=Gamma1,
                                Gamma2=Gamma2, sign=1)
        self.get_rho_steady_state()
        if not self.Lindbladian.super_fermi_ops.tilde_conjugationrule_phase:
            self.Lindbladian.update(T_mat=T_mat, U_mat=U_mat, Gamma1=Gamma1,
                                    Gamma2=Gamma2, sign=-1)
        self.sectors_exact_decomposition()
        self.reset_correlator_data()

    def get_operator_keys(self, contour_parameters, operator_sectors):

        operator_sector_list = [operator_sectors[p[2]][p[3]] if p[2] != 'rho'
                                else add_sectors(operator_sectors['cdag'][
                                    p[3]],
                                operator_sectors['c'][p[4]])
                                for p in contour_parameters]
        sectors = [(0, 0), *operator_sector_list, (0, 0)]

        operators = []
        for p in contour_parameters:
            tmp = None
            if p[1] >= 0:
                if p[2] == 'rho':
                    tmp = (p[2], p[3], p[4])
                else:
                    tmp = (p[2], p[3])
            else:
                if p[2] == 'rho':
                    tmp = (p[2] + "_tilde", p[3], p[4])
                else:
                    tmp = (
                        self.Lindbladian.super_fermi_ops.tilde_operator_name[
                            p[2]], p[3])
            operators.append(tmp)
        tmp = (0, 0)
        spin_sector_transitions = []
        for s in sectors[::-1][1:]:
            sum = add_sectors(s, tmp)
            spin_sector_transitions.append((sum, tmp))
            tmp = sum
        spin_sector_transitions = spin_sector_transitions[::-1][1:]
        operator_keys = tuple(map(lambda x: (*x[0], x[1]),
                                  zip(operators, spin_sector_transitions)))
        assert spin_sector_transitions[0][0] == (0, 0), \
            ("The number of creation and annihilation have to be the same per"
             + "spin.")
        return operator_keys

    def precalculate_expectation_value(self, sites, operator_keys, shape,
                                       n_fop, n_time):
        """Calculate the expectation value of a given set of operators with
        regards to the steady state density of states and at time zero. It is
        calculated by insertion of the Lindbladian eigenbasis and saved as a
        separately, e.g.
        <I|A|R_n><L_n|B|R_m><L_n|C|rho>=:D_nm  with n in {0,...,N} and m in
        {0,...,M}
        The operators A,B,C can be a collection fermionic operators.

        This can later be used to determine the time/frequency dependend
        expectattion values.

        Parameters
        ----------
        operator_keys : tuple
            keys of the operators containing the sectors, the spin, the
        shape : tuple of ints
            containing the dimension of the expectation value
        n_fop : int
            number of fermionic creation and annihilation operators
        n_time : int
            number of operators A,B,C
        """
        # precalculate the needed terms for calculating the corresponding
        # correlation function
        if operator_keys not in self.precalc_correlator[n_fop].keys():
            spin_sector_fermi_ops = []
            vec_r_sector = []
            vec_l_sector = []
            if n_time < 5:
                for site, op_key in zip(sites, operator_keys):
                    tmp = None
                    if op_key[0] != 'rho' and op_key[0] != 'rho_tilde':
                        tmp = ((self.Lindbladian.super_fermi_ops
                                ).spin_sector_fermi_ops[site][op_key[0]]
                               [op_key[1]][op_key[2]].todense() *
                               (self.Lindbladian.super_fermi_ops
                                ).tilde_operator_sign[op_key[0]])
                    else:
                        if op_key[0] == 'rho':
                            middle_sector = add_sectors(
                                self.operator_sectors['c'][op_key[2]],
                                op_key[3][1])
                            tmp = (
                                (self.Lindbladian.super_fermi_ops
                                 ).spin_sector_fermi_ops[site[0]]['cdag'][
                                     op_key[1]][op_key[3][0], middle_sector]
                            ).dot((self.Lindbladian.super_fermi_ops
                                   ).spin_sector_fermi_ops[site[1]]['c']
                                  [op_key[2]][middle_sector, op_key[3][1]]
                                  )
                        elif op_key[0] == 'rho_tilde':
                            middle_sector = add_sectors(
                                self.operator_sectors['cdag'][op_key[1]],
                                op_key[3][1])
                            tmp = (
                                (self.Lindbladian.super_fermi_ops
                                 ).spin_sector_fermi_ops['cdag_tilde']
                                [op_key[2]][op_key[3][0], middle_sector]).dot(
                                (self.Lindbladian.super_fermi_ops
                                 ).spin_sector_fermi_ops['c_tilde']
                                [op_key[1]][middle_sector, op_key[3][1]]
                            )
                    spin_sector_fermi_ops.append(tmp)
                spin_sector_fermi_ops = tuple(spin_sector_fermi_ops)
                vec_r_sector = tuple(
                    [self.vec_r_sector[
                        op_key[2][0]] if (
                            op_key[0] != 'rho' and op_key[0] != 'rho_tilde')
                        else self.vec_r_sector[op_key[3][0]] for op_key in
                        operator_keys[1:]])
                vec_l_sector = tuple(
                    [self.vec_l_sector[
                        op_key[2][0]] if (
                            op_key[0] != 'rho' and op_key[0] != 'rho_tilde')
                        else self.vec_l_sector[op_key[3][0]] for op_key in
                        operator_keys[1:]])

            if n_time == 2:
                self.precalc_correlator[n_time][operator_keys] = \
                    _precalculate_two_point_correlator(
                    shape, self.left_vacuum,
                    spin_sector_fermi_ops,
                    vec_r_sector,
                    vec_l_sector,
                    self.rho_stready_state, n_time)
            elif n_time == 3:
                if self.trilex:
                    self.precalc_correlator['trilex'][operator_keys] = \
                        _precalculate_three_point_correlator(
                        shape, self.left_vacuum,
                        spin_sector_fermi_ops,
                        vec_r_sector,
                        vec_l_sector,
                        self.rho_stready_state, n_time)
                else:
                    self.precalc_correlator[n_time][operator_keys] = \
                        _precalculate_three_point_correlator(
                        shape, self.left_vacuum,
                        spin_sector_fermi_ops,
                        vec_r_sector,
                        vec_l_sector,
                        self.rho_stready_state, n_time)
            elif n_time == 4:
                self.precalc_correlator[n_time][operator_keys] = \
                    _precalculate_four_point_correlator(
                    shape, self.left_vacuum,
                    spin_sector_fermi_ops,
                    vec_r_sector,
                    vec_l_sector,
                    self.rho_stready_state, n_time)

    def get_single_particle_green(self, component, freq, sites=None,
                                  spin=('up', 'up')):
        if sites is None:
            site = int(
                (self.Lindbladian.super_fermi_ops.fock_ops.nsite - 1) / 2)
            sites = (site, site)
        permutation_sign = None
        operators = None

        if component == (1, 0):
            permutation_sign = 1 + 0j
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((0, 0, 'cdag', spin[1]),
                                         (1, -1, 'c', spin[0])),
                                        ((1, 0, 'c', spin[0]),
                                         (0, 0, 'cdag', spin[1]))]]

        elif component == (0, 1):
            permutation_sign = -1 + 0j
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((1, 0, 'cdag', spin[1]),
                                         (0, 0, 'c', spin[0])),
                                        ((0, 0, 'c', spin[0]),
                                         (1, -1, 'cdag', spin[1]))]]

        tensor_shapes = [tuple([self.vals_sector[op_key[2][1]].shape[0]
                                if op_key[0] != 'rho' else
                                self.vals_sector[op_key[3][1]].shape[0]
                                for op_key in op_keys[:-1]]) for op_keys
                         in operators]
        vals_sector = tuple([tuple([self.vals_sector[op_key[2][0]] if
                            (op_key[0] != 'rho' and op_key[0] != 'rho_tilde')
                            else self.vals_sector[op_key[3][0]]
                            for op_key in op_keys[1:]])[0] for op_keys
            in operators])

        n_fop = 2
        n_time = 2
        precalc_correlator = []
        for op_key, tensor_shape in zip(operators, tensor_shapes):
            self.precalculate_expectation_value(sites, op_key, tensor_shape,
                                                n_fop, n_time)
            precalc_correlator.append(self.precalc_correlator[2][op_key])
        precalc_correlator = tuple(precalc_correlator)

        green_component_plus = np.zeros(freq.shape, dtype=np.complex128)
        green_component_minus = np.zeros(freq.shape, dtype=np.complex128)
        if component == (1, 0):
            _get_two_point_correlator_frequency(green_component_plus,
                                                green_component_minus, freq,
                                                precalc_correlator,
                                                vals_sector,
                                                tensor_shapes,
                                                n_fop, permutation_sign)
        elif component == (0, 1):
            _get_two_point_correlator_frequency(green_component_plus,
                                                green_component_minus, freq,
                                                precalc_correlator,
                                                vals_sector,
                                                tensor_shapes,
                                                n_fop, permutation_sign)
        return green_component_plus, green_component_minus

    def get_three_point_vertex_components(
            self, component, freq, sites=None, spin=('up', 'up', ('up', 'up')),
            permutation_sign=(-1 + 0j, 1 + 0j, 1 + 0j), prefactor=-1 + 0j):
        operators = None
        if sites is None:
            site = int(
                (self.Lindbladian.super_fermi_ops.fock_ops.nsite - 1) / 2)
            sites = (site, site, (site, site))
        # precalculating the expectation value
        if component == (0, 0, 0):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((0, 0, 'c', spin[0]),
                                         (0, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0],
                                         spin[2][1])),
                                        ((0, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0],
                                         spin[2][1]),
                                         (0, 0, 'cdag', spin[1])),

                                        ((0, 0, 'cdag', spin[1]), (0, 0, 'c',
                                                                   spin[0]),
                                         (0, 0, 'rho', spin[2][0],
                                         spin[2][1])),
                                        ((0, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0])),

                                        ((0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0]), (0, 0, 'cdag',
                                                                spin[1])),
                                        ((0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'cdag', spin[1]), (0, 0, 'c',
                                                                   spin[0]))]]

        elif component == (1, 0, 0):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((1, 0, 'c', spin[0]),
                                         (0, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0],
                                         spin[2][1])),
                                        ((0, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, -1, 'c', spin[0])),

                                        ((1, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'cdag', spin[1])),
                                        ((0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'cdag', spin[1]), (1, -1, 'c',
                                         spin[0]))]]

        elif component == (0, 1, 0):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((1, 0, 'cdag', spin[1]),
                                         (0, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0],
                                         spin[2][1])),
                                        ((0, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, -1, 'cdag', spin[1])),

                                        ((1, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0])),
                                        ((0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0]), (1, -1, 'cdag',
                                                                spin[1]))]]

        elif component == (0, 0, 1):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0]),
                                         (0, 0, 'cdag', spin[1])),
                                        ((0, 0, 'c', spin[0]),
                                        (0, 0, 'cdag', spin[1]),
                                         (1, -1, 'rho', spin[2][0],
                                         spin[2][1])),

                                        ((1, 0, 'rho', spin[2][0],
                                          spin[2][1]),
                                         (0, 0, 'cdag', spin[1]),
                                         (0, 0, 'c', spin[0])),
                                        ((0, 0, 'cdag', spin[1]),
                                        (0, 0, 'c', spin[0]),
                                         (1, -1, 'rho', spin[2][0],
                                         spin[2][1]))]]

        elif component == (1, 1, 0):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((1, 0, 'c', spin[0]),
                                         (1, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0],
                                          spin[2][1])),
                                        ((1, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, -1, 'c', spin[0])),
                                        ((1, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, -1, 'c', spin[0])),

                                        ((1, 0, 'cdag', spin[1]),
                                         (1, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0],
                                            spin[2][1])),
                                        ((1, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, -1, 'cdag', spin[1])),
                                        ((1, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, -1, 'cdag', spin[1]))]]

        elif component == (1, 0, 1):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((1, 0, 'c', spin[0]),
                                         (1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'cdag', spin[1])),
                                        ((1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'cdag', spin[1]),
                                         (1, -1, 'c', spin[0])),
                                        ((1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'cdag', spin[1]),
                                         (1, -1, 'c', spin[0])),

                                        ((1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, 0, 'c', spin[0]),
                                         (0, 0, 'cdag', spin[1])),
                                        ((1, 0, 'c', spin[0]),
                                        (0, 0, 'cdag', spin[1]),
                                         (1, -1, 'rho', spin[2][0],
                                         spin[2][1])),
                                        ((1, 0, 'c', spin[0]),
                                        (0, 0, 'cdag', spin[1]),
                                         (1, -1, 'rho', spin[2][0],
                                         spin[2][1]))]]

        elif component == (0, 1, 1):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((1, 0, 'cdag', spin[1]),
                                         (1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0])),
                                        ((1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0]),
                                         (1, -1, 'cdag', spin[1])),
                                        ((1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0]),
                                         (1, -1, 'cdag', spin[1])),

                                        ((1, 0, 'rho', spin[2][0], spin[2][1]),
                                         (1, 0, 'cdag', spin[1]),
                                         (0, 0, 'c', spin[0])),
                                        ((1, 0, 'cdag', spin[1]),
                                        (0, 0, 'c', spin[0]),
                                         (1, -1, 'rho', spin[2][0],
                                         spin[2][1])),
                                        ((1, 0, 'cdag', spin[1]),
                                        (0, 0, 'c', spin[0]),
                                         (1, -1, 'rho', spin[2][0],
                                         spin[2][1]))]]

        elif component == (1, 1, 1):
            operators = [self.get_operator_keys(op_key, self.operator_sectors)
                         for op_key in [((0, 0, 'cdag', spin[1]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, -1, 'c', spin[0])),
                                        ((0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'cdag', spin[1]),
                                         (0, -1, 'c', spin[0])),

                                        ((0, 0, 'c', spin[0]),
                                         (0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, -1, 'cdag', spin[1])),
                                        ((0, 0, 'rho', spin[2][0], spin[2][1]),
                                         (0, 0, 'c', spin[0]),
                                         (0, -1, 'cdag', spin[1])),

                                        ((0, 0, 'c', spin[0]),
                                        (0, 0, 'cdag', spin[1]),
                                         (0, -1, 'rho', spin[2][0],
                                         spin[2][1])),
                                        ((0, 0, 'cdag', spin[1]),
                                        (0, 0, 'c', spin[0]),
                                         (0, -1, 'rho', spin[2][0],
                                         spin[2][1]))]]

        tensor_shapes = [tuple([self.vals_sector[op_key[2][1]].shape[0]
                                if op_key[0] != 'rho' else
                                self.vals_sector[op_key[3][1]].shape[0]
                                for op_key in op_keys[:-1]]) for op_keys
                         in operators]
        vals_sectors = [
            tuple([self.vals_sector[op_key[2][0]] if
                   (op_key[0] != 'rho' and op_key[0] != 'rho_tilde')
                   else self.vals_sector[op_key[3][0]]
                   for op_key in op_keys[1:]]) for op_keys
            in operators]

        n_fop = 4
        n_time = 3
        precalc_correlators = []
        for op_key, tensor_shape in zip(operators, tensor_shapes):
            self.precalculate_expectation_value(op_key, tensor_shape,
                                                n_fop, n_time)
            precalc_correlators.append(
                self.precalc_correlator['trilex'][op_key])
        precalc_correlators = tuple(precalc_correlators)

        green_component = np.zeros((freq.shape[0], freq.shape[0]),
                                   dtype=np.complex128)

        if component == (0, 0, 0):
            _get_three_point_correlator_frequency_mmm(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        elif component == (1, 0, 0):
            _get_three_point_correlator_frequency_pmm(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        elif component == (0, 1, 0):
            _get_three_point_correlator_frequency_mpm(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        elif component == (0, 0, 1):
            _get_three_point_correlator_frequency_mmp(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        elif component == (1, 1, 0):
            _get_three_point_correlator_frequency_ppm(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        elif component == (1, 0, 1):
            _get_three_point_correlator_frequency_pmp(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        elif component == (0, 1, 1):
            _get_three_point_correlator_frequency_mpp(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        elif component == (1, 1, 1):
            _get_three_point_correlator_frequency_ppp(green_component, freq,
                                                      precalc_correlators,
                                                      vals_sectors,
                                                      tensor_shapes,
                                                      permutation_sign,
                                                      prefactor)
        return green_component

    def get_three_point_vertex(self, freq,
                               spin=('up', 'up', ('up', 'up')),
                               permutation_sign=(-1 + 0j, 1 + 0j, 1 + 0j),
                               prefactor=-1 + 0j,
                               return_=False):
        if not return_:

            for component in self.correlators['trilex'][spin]:
                self.correlators['trilex'][spin][component] = \
                    self.get_three_point_vertex_components(
                        component, freq, permutation_sign=permutation_sign,
                        prefactor=prefactor)
        else:
            three_point_vertex = np.zeros(
                (freq.shape[0], freq.shape[0], 2, 2, 2),
                dtype=np.complex128)
            for i, j, k in self.correlators['trilex'][
                    ("up", "up", ("up", "up"))]:
                three_point_vertex[:, :, i, j, k] = \
                    self.get_three_point_vertex_components(
                        (i, j, k), freq, spin,
                    permutation_sign=permutation_sign, prefactor=prefactor)

            return three_point_vertex

    def set_contour_symmetries(self, n_correlators, trilex=False):
        self.contour_symmetries = {}
        n_corr = n_correlators.copy()
        if trilex and 3 not in n_corr:
            n_corr.append(3)

        for n in n_corr:
            contour = get_branch_combinations(n)
            t_max = t_max_possitions(n)
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
                    t_max_idx = find_index_max_time(comb)

                    if comb[t_max_idx][0] == 1:
                        n_correlator_symmetrie[comb] = None
                        comb_symmetric = list(comb)
                        comb_symmetric[t_max_idx] = tuple(
                            [x if i != 0 else 0 for i, x in
                             enumerate(comb[t_max_idx])])
                        n_correlator_symmetrie[tuple(comb_symmetric)] = comb
            if n == 2:
                choose_components([((1, 1), (0, 0)), ((1, 0), (0, 1)),
                                   ((0, 1), (1, 0)), ((0, 0), (1, 1))],
                                  n_correlator_symmetrie)
            if n == 3 and trilex:
                self.contour_symmetries['trilex'] = \
                    n_correlator_symmetrie.copy()
            else:
                self.contour_symmetries[n] = n_correlator_symmetrie.copy()


@njit(parallel=True, cache=True)
def _get_two_point_correlator_frequency(green_component_plus,
                                        green_component_minus, freq,
                                        precalc_correlators, vals_sectors,
                                        tensor_shapes,
                                        n_operator, permutation_sign):

    for i in prange(len(freq)):
        G_plus = 0 + 0j
        G_minus = 0 + 0j

        for n in prange(tensor_shapes[1][0]):
            G_plus += (precalc_correlators[1][n] /
                       ((1j * freq[i] + vals_sectors[1][n])))

        for n in prange(tensor_shapes[0][0]):
            G_minus += (precalc_correlators[0][n] /
                        ((1j * freq[i] - vals_sectors[0][n])))

        green_component_plus[i] = 1j * G_plus * permutation_sign
        green_component_minus[i] = -1j * G_minus * permutation_sign


# @njit(parallel=True, cache=True)
# def _get_two_point_correlator_frequency_mp(green_component, freq,
#                                            precalc_correlators, vals_sectors,
#                                            tensor_shapes,
#                                            n_operator, permutation_sign):

#     for i in prange(len(freq)):
#         G1 = 0 + 0j
#         G2 = 0 + 0j
#         for n in prange(tensor_shapes[0][0]):
#             G1 += (precalc_correlators[0][n] /
#                    ((1j * freq[i] - vals_sectors[0][n])))
#         for n in prange(tensor_shapes[1][0]):
#             G2 += (precalc_correlators[1][n] /
#                    ((1j * freq[i] + vals_sectors[1][n])))

#         green_component[i] = (G1 - G2) * \
#             ((-1j)) * permutation_sign
#     # return green_component


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_mmm(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign, prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor * precalc_correlators[0][n, m] * (1.0 / (
                        (1j * w1 + L_n) * (1j * (w1 + w2) + L_n + L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * permutation_sign[2]\
                        * precalc_correlators[1][n, m] * (-1 / (
                            (1j * w1 + L_n) * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * permutation_sign[0]\
                        * precalc_correlators[2][n, m] * (1 / (
                            (1j * (w1 + w2) + L_n + L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[0] * permutation_sign[1]\
                        * precalc_correlators[3][n, m] * (-1 / (
                            (1j * w1 - L_n - L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                for m in prange(tensor_shapes[4][1]):
                    L_m = vals_sectors[4][1][m]
                    G += prefactor * permutation_sign[1] * permutation_sign[2]\
                        * precalc_correlators[4][n, m] * (1 / (
                            (1j * (w1 + w2) - L_n) * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                for m in prange(tensor_shapes[5][1]):
                    L_m = vals_sectors[5][1][m]
                    G += prefactor * permutation_sign[1] \
                        * permutation_sign[2] * permutation_sign[0] \
                        * precalc_correlators[5][n, m]\
                        * (1 / ((1j * w1 - L_n - L_m) * (1j * (w1 + w2) - L_n))
                           )

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_pmm(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign,
                                              prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor * precalc_correlators[0][n, m] * (1.0 / (
                        (1j * w1 + L_n) * (1j * w2 + L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * precalc_correlators[1][n, m] * (-1 / (
                        (1j * w1 - L_n - L_m) * (1j * w2 + L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * permutation_sign[2]\
                        * precalc_correlators[2][n, m] * (-1 / (
                            (1j * w1 + L_n) * (1j * (w1 + w2) - L_m)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[2]\
                        * precalc_correlators[3][n, m] * (1 / (
                            (1j * w1 - L_n - L_m) * (1j * (w1 + w2) - L_n)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_mpm(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign,
                                              prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor * permutation_sign[0]\
                        * precalc_correlators[0][n, m] * (1.0 / (
                            (1j * w1 + L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * permutation_sign[0]\
                        * precalc_correlators[1][n, m] * (-1 / (
                            (1j * w1 + L_n) * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * permutation_sign[0]\
                        * permutation_sign[1]\
                        * precalc_correlators[2][n, m] * (-1 / (
                            (1j * (w1 + w2) - L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[0]\
                        * permutation_sign[1]\
                        * precalc_correlators[3][n, m] * (1 / (
                            (1j * w2 - L_n - L_m) * (1j * (w1 + w2) - L_n)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_mmp(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign,
                                              prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor * permutation_sign[1]\
                        * permutation_sign[2]\
                        * precalc_correlators[0][n, m] * (-1 / (
                            (1j * w1 + L_m) * (1j * (w1 + w2) - L_n)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * permutation_sign[1]\
                        * permutation_sign[2]\
                        * precalc_correlators[1][n, m] * (1 / (
                            (1j * w1 + L_n) * (1j * (w1 + w2) + L_n + L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * permutation_sign[1]\
                        * permutation_sign[2] * permutation_sign[0]\
                        * precalc_correlators[2][n, m] * (-1 / (
                            (1j * w2 + L_m) * (1j * (w1 + w2) - L_n)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[1]\
                        * permutation_sign[2] * permutation_sign[0]\
                        * precalc_correlators[3][n, m] * (1 / (
                            (1j * w2 + L_n) * (1j * (w1 + w2) + L_n + L_m)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_ppm(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign,
                                              prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor * precalc_correlators[0][n, m] * (1.0 / (
                        (1j * (w1 + w2) + L_n + L_m) * (1j * w2 + L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * precalc_correlators[1][n, m] * (-1 / (
                        (1j * w1 - L_n - L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * precalc_correlators[2][n, m]\
                        * ((1 / (1j * w1 - L_n - L_m))
                           - (1 / (1j * (w1 + w2) - L_m))
                           ) * (1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[0]\
                        * precalc_correlators[3][n, m]\
                        * ((1 / (1j * w1 + L_m))
                           - (1 / (1j * (w1 + w2) + L_n + L_m))
                           ) * (1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                for m in prange(tensor_shapes[4][1]):
                    L_m = vals_sectors[4][1][m]
                    G += prefactor * permutation_sign[0]\
                        * precalc_correlators[4][n, m] * (-1 / (
                            (1j * (w1 + w2) - L_m) * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                for m in prange(tensor_shapes[5][1]):
                    L_m = vals_sectors[5][1][m]
                    G += prefactor * permutation_sign[0]\
                        * precalc_correlators[5][n, m] * (1 / (
                            (1j * (w1 + w2) - L_m) * (1j * w2 - L_n - L_m)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_pmp(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign,
                                              prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor * permutation_sign[2]\
                        * precalc_correlators[0][n, m] * (1 / (
                            (1j * (w1 + w2) - L_m) * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * permutation_sign[2]\
                        * precalc_correlators[1][n, m]\
                        * ((1 / (1j * w1 - L_n - L_m))
                           - (1 / (1j * (w1 + w2) - L_n))
                           ) * (1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * permutation_sign[2]\
                        * precalc_correlators[2][n, m]\
                        * (1 / (1j * w1 - L_n - L_m)) * (-1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[2] * permutation_sign[1]\
                        * precalc_correlators[3][n, m]\
                        * (-1 / (1j * w1 + L_m)) * (1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                for m in prange(tensor_shapes[4][1]):
                    L_m = vals_sectors[4][1][m]
                    G += prefactor * permutation_sign[2] * permutation_sign[1]\
                        * precalc_correlators[4][n, m]\
                        * ((1 / (1j * w1 + L_n))
                           - (1 / (1j * (w1 + w2) + L_n + L_m))
                           ) * (1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                for m in prange(tensor_shapes[5][1]):
                    L_m = vals_sectors[5][1][m]
                    G += prefactor * permutation_sign[2] * permutation_sign[1]\
                        * precalc_correlators[5][n, m] * (1 / (
                            (1j * (w1 + w2) + L_n + L_m) * (1j * w2 + L_m)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_mpp(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign,
                                              prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor * permutation_sign[0] \
                        * permutation_sign[1] * precalc_correlators[0][n, m]\
                        * ((1 / (1j * w1 - L_n - L_m))
                           - (1 / (1j * (w1 + w2) - L_m))
                           ) * (1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * permutation_sign[0] * permutation_sign[1]\
                        * precalc_correlators[1][n, m]\
                        * (1 / (1j * (w1 + w2) - L_n)) * (
                            1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * permutation_sign[0] * permutation_sign[1]\
                        * precalc_correlators[2][n, m]\
                        * (-1 / (1j * w1 + L_m)) * (1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[0] * permutation_sign[1]\
                        * permutation_sign[2] * precalc_correlators[3][n, m]\
                        * (1 / (1j * w1 - L_n - L_m)) * (-1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                for m in prange(tensor_shapes[4][1]):
                    L_m = vals_sectors[4][1][m]
                    G += prefactor * permutation_sign[0] * permutation_sign[1]\
                        * permutation_sign[2] * precalc_correlators[4][n, m]\
                        * (1 / (1j * (w1 + w2) + L_n + L_m)) * (
                            1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                for m in prange(tensor_shapes[5][1]):
                    L_m = vals_sectors[5][1][m]
                    G += prefactor * permutation_sign[0] * permutation_sign[1]\
                        * permutation_sign[2] * precalc_correlators[5][n, m]\
                        * ((1 / (1j * w1 + L_m)) - (
                            1 / (1j * (w1 + w2) + L_n + L_m))
                           ) * (1 / (1j * w2 + L_n))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _get_three_point_correlator_frequency_ppp(green_component, freq,
                                              precalc_correlators,
                                              vals_sectors,
                                              tensor_shapes,
                                              permutation_sign,
                                              prefactor):
    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                for m in range(tensor_shapes[0][1]):
                    L_m = vals_sectors[0][1][m]
                    G += prefactor\
                        * precalc_correlators[0][n, m]\
                        * (1 / (1j * w1 - L_n - L_m)) * (
                            1 / (1j * (w1 + w2) - L_m))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                for m in prange(tensor_shapes[1][1]):
                    L_m = vals_sectors[1][1][m]
                    G += prefactor * permutation_sign[2]\
                        * precalc_correlators[1][n, m]\
                        * (1 / (1j * w1 - L_n - L_m)) * (-1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                for m in prange(tensor_shapes[2][1]):
                    L_m = vals_sectors[2][1][m]
                    G += prefactor * permutation_sign[0]\
                        * precalc_correlators[2][n, m]\
                        * (1 / (1j * (w1 + w2) - L_m)) * (
                            1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                for m in prange(tensor_shapes[3][1]):
                    L_m = vals_sectors[3][1][m]
                    G += prefactor * permutation_sign[0] \
                        * permutation_sign[1] * precalc_correlators[3][n, m]\
                        * (-1 / (1j * w1 + L_m)) * (
                            -1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                for m in prange(tensor_shapes[4][1]):
                    L_m = vals_sectors[4][1][m]
                    G += prefactor * permutation_sign[1] * permutation_sign[2]\
                        * precalc_correlators[4][n, m]\
                        * (1 / (1j * (w1 + w2) + L_n + L_m)) * (
                            1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                for m in prange(tensor_shapes[5][1]):
                    L_m = vals_sectors[5][1][m]
                    G += prefactor * permutation_sign[0] * permutation_sign[1]\
                        * permutation_sign[2] * precalc_correlators[5][n, m]\
                        * (1 / (1j * w1 + L_m)) * (
                            1 / (1j * (w1 + w2) + L_n + L_m))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def _precalculate_two_point_correlator(shape, left_vacuum,
                                       spin_sector_fermi_ops, vec_r_sector,
                                       vec_l_sector, rho_stready_state, n):
    assert n == 2
    precalc_corr_tmp = np.zeros(shape, dtype=np.complex128)

    for i in prange(shape[0]):
        precalc_corr_tmp[i] = left_vacuum.dot(spin_sector_fermi_ops[0]).dot(
            vec_r_sector[0][i]).dot(
            vec_l_sector[0][i]).dot(
            spin_sector_fermi_ops[1]).dot(
            rho_stready_state)[0, 0]

    return precalc_corr_tmp


@njit(parallel=True, cache=True)
def _precalculate_three_point_correlator(shape, left_vacuum,
                                         spin_sector_fermi_ops, vec_r_sector,
                                         vec_l_sector, rho_stready_state, n):
    assert n == 3
    precalc_corr_tmp = np.zeros(shape, dtype=np.complex128)

    expectation_start = left_vacuum.dot(spin_sector_fermi_ops[0])

    for i in prange(shape[0]):
        expectation_val = expectation_start.dot(
            vec_r_sector[0][i]).dot(
            vec_l_sector[0][i]).dot(
            spin_sector_fermi_ops[1])

        for j in prange(shape[1]):
            precalc_corr_tmp[i, j] = expectation_val.dot(
                vec_r_sector[1][j]).dot(
                vec_l_sector[1][j]).dot(
                spin_sector_fermi_ops[2]).dot(
                rho_stready_state)[0, 0]

    return precalc_corr_tmp


@njit(parallel=True, cache=True)
def _precalculate_four_point_correlator(shape, left_vacuum,
                                        spin_sector_fermi_ops, vec_r_sector,
                                        vec_l_sector, rho_stready_state, n):
    assert n == 4
    precalc_corr_tmp = np.zeros(shape, dtype=np.complex128)

    expectation_start = left_vacuum.dot(spin_sector_fermi_ops[0])

    for i in prange(shape[0]):
        expectation_val_1 = expectation_start.dot(
            vec_r_sector[0][i]).dot(
            vec_l_sector[0][i]).dot(
            spin_sector_fermi_ops[1])

        for j in prange(shape[1]):
            expectation_val_2 = expectation_val_1.dot(
                vec_r_sector[1][j]).dot(
                vec_l_sector[1][j]).dot(
                spin_sector_fermi_ops[2])
            for k in prange(shape[2]):
                precalc_corr_tmp[i, j, k] = expectation_val_2.dot(
                    vec_r_sector[2][k]).dot(
                    vec_l_sector[2][k]).dot(
                    spin_sector_fermi_ops[3]).dot(
                    rho_stready_state)[0, 0]

    return precalc_corr_tmp


def choose_components(components, contour_symmetries):
    for key in contour_symmetries:
        value = contour_symmetries[key]
        if key in components:
            if value is not None:
                contour_symmetries[value] = key
                contour_symmetries[key] = None


def get_branch_combinations(n, contour_ordered=False):
    # forward branch 0/ backward branch 1
    # Returns all possible branch combinations
    combos = []
    for i in range(2**n):
        compination = list(map(int, bin(i).replace("0b", "")))
        padding = [0 for i in range(n - len(compination))]
        combos.append(padding + compination)
    combos = np.array(combos)
    if contour_ordered:
        mask = list(map(lambda x: np.all(
            x == sorted(x, reverse=True)), combos))
        return list(map(lambda x: tuple(x), combos[mask]))
    return list(map(lambda x: tuple(x), combos))


def add_sectors(sector1, sector2):
    return tuple([sum(x) for x in zip(sector1, sector2)])


def t_max_possitions(n):
    t_max = list(map(lambda x: tuple(x), itertools.permutations(
        [0 if i != n - 1 else 1 for i in range(n)], r=n)))
    return set(t_max)


def find_index_max_time(list_of_parameters):
    return list_of_parameters.index(
        max(list_of_parameters, key=(lambda x: x[1])))


# %%
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # Set parameters
    ws = np.linspace(-10, 10, 200)

    Nb = 1
    nsite = 2 * Nb + 1
    U_imp = 0.0
    es = np.array([1])
    ts = np.array([0.5])
    Us = np.zeros(nsite)
    Us[Nb] = U_imp
    gamma = np.array([0.2 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])

    # Initializing Lindblad class
    spin_sector_max = 1
    spinless = True
    tilde_conjugationrule_phase = False
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

# %%
