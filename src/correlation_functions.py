"""Here the base code for the construction of correlation functions
    will be written.
    """
# %%
import enum
import itertools
import numpy as np
from numba import njit
from sympy import li, symbols, Matrix
import re
import src.model_lindbladian as lind
import src.lindbladian_exact_decomposition as ed_lind
import src.auxiliary_system_parameter as aux
import src.frequency_greens_function as fg
from src.dos_util import heaviside
import matplotlib.pyplot as plt

# def get_index(args...):
#     indices = np.zeros((np.prod(args),len(args)))
#     idx = np.zeros(len(args))
#     for a in args:
#         for i in range(a)


class Correlators:
    def __init__(self, Lindbladian, spin_sector_max, correlators=None):
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

        spin_sector_max : int
            maximal value of a spin difference between "normal" and "tilde"
            space
        """
        assert spin_sector_max >= 0
        self.Lindbladian = Lindbladian
        self.nsite = self.Lindbladian.liouville_ops.fock_ops.nsite
        self.spin_sector_max = spin_sector_max
        self.set_spin_sectors()
        self.set_spin_sectors_fermionic_ops()
        self.operator_sectors = {"cdag": {"up": (1, 0), 'do': (0, 1)},
                                 'c': {'up': (-1, 0), 'do': (0, -1)}}

        if correlators is None:
            correlators = [2*i for i in range(1, spin_sector_max+1)]

        self.permutation_signs = {c: get_permutations_sign(c) for c in
        correlators}

        self.precalc_correlator = {n: {} for n in correlators}

        self.set_contour_symmetries(correlators)
        self.correlators = {n: None for n in correlators}

    def update_model_parameter(self, Gamma1, Gamma2, T_mat, U_mat=None):
        """Update the model parameter, in order to recalculate the Lindbladian.

        Parameters
        ----------
        Gamma1 : numpy.ndarray (dim,dim)
            2D array with the coupling to bath 1, describing the removal of
            electrons.

        Gamma2 : numpy.ndarray (dim,dim)
            2D array with the coupling to bath 2, describing the injection of
            electrons.

        T_mat : numpy.ndarray (dim,dim)
            Hopping matrix.

        U_mat : numpy.ndarray 1D or 2D, optional
            On-site interaction strength. It can differ at each site,
            by default None.
        """
        if T_mat is not None:
            self.T_mat = T_mat
        if T_mat is not None:
            self.U_mat = U_mat
        self.Gamma1 = Gamma1
        self.Gamma2 = Gamma2

    def set_lindbladian(self, sign):
        """Set up the Lindbladian

        Parameters
        ----------
        sign : int (-1 or 1)
            -1 if the Lindbladian describes the time evolution of a single
            fermionic operator and with a dissipator describing a single
            electron dissipation, 1 otherwise.
        """
        assert np.abs(sign) == 1
        self.sign = sign
        self.Lindbladian.set_unitay_part(self.T_mat, self.U_mat)
        self.Lindbladian.set_dissipation(self.Gamma1, self.Gamma2, sign)
        self.Lindbladian.set_total_linbladian()
        self.precalc_expectation_value = {}


    def set_spin_sectors(self):
        """calculate all relevant spin sectors, that can be reached with the
        highest, desired correlator. Set all possible sectors as attribute of
        the object.

        E.G. a four point vertex can access all spin sector , (i,j)
        with i<=2, j<=2 and |i-j|<=2

        """
        sector_range = np.arange(-self.spin_sector_max,
                                 self.spin_sector_max + 1)
        spin_combination = np.array(
            np.meshgrid(sector_range, sector_range)).T.reshape(-1, 2)
        allowed_combinations = np.sum(np.abs(spin_combination), -1)
        allowed_combinations = allowed_combinations <= self.spin_sector_max
        self.spin_combination = spin_combination[allowed_combinations]
        self.spin_combination = list(map(lambda x: tuple(x),
                                         self.spin_combination))
        self.projectors = {}
        for com in self.spin_combination:
            self.projectors[tuple(com)] = (self.Lindbladian.liouville_ops
                                           ).spin_sector_permutation_operator(
                                               com)

    def get_operator_in_spin_sector(self, Object, sector_left=None,
                                    sector_right=None):
        """Extract an Liouville space operator or vector in the desired spin
        sector subspace.

        Parameters
        ----------
        Object : scipy.sparse.csc_matrix (dim, 1), (1, dim) or (dim, dim)
            Liouville space operator or vector.

        sector_left : tuple (int, int), optional
            Left spin sector, by default None.

        sector_right : tuple (int, int), optional
            Right spin sector, by default None.

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Object in subspace connecting, which connects the spin sectors
            sector_left and sector_right.

        Raises
        ------
        ValueError
            If no sector_left is passed, Object has to be a vector of shape
            (1, dim).
        ValueError
            "A Sector has to be passed."
        ValueError
            If no sector_right is passed, Object has to be a vector of shape
            (dim, 1).
        """
        if sector_left is None:
            if Object.shape[0] != 1:
                raise ValueError("If no sector_left is passed, Object has" +
                                 " to be a vector of shape (1, dim).")
            if sector_right is None:
                raise ValueError("A Sector has to be passed.")
            return (Object * self.projectors[sector_right][1].transpose()
                    )[0, :self.projectors[sector_right][0]]
        elif sector_right is None:
            if Object.shape[1] != 1:
                raise ValueError("If no sector_right is passed, Object has" +
                                 " to be a vector of shape (dim, 1).")
            return (self.projectors[sector_left][1] * Object
                    )[:self.projectors[sector_left][0], 0]
        else:
            return (self.projectors[sector_left][1] * Object
                    * self.projectors[sector_right][1].transpose()
                    )[:self.projectors[sector_left][0],
                      :self.projectors[sector_right][0]]

    def set_rho_steady_state(self):
        """Calculate the steady state density of state in the spin sector
        (0,0).

        Raises
        ------
        ValueError
            If there are more then one steady-steate density of states
        """
        self.set_lindbladian(1.0)
        L_00 = self.get_operator_in_spin_sector(self.Lindbladian.L_tot, (0, 0),
                                                (0, 0))

        vals, _, vec_r = ed_lind.exact_spectral_decomposition(
            L_00.todense())

        mask = np.isclose(vals, np.zeros(vals.shape))
        vals_close_zero = vals[mask]
        n_steady_state = vals_close_zero.shape[0]
        if n_steady_state >1:
            vals_close_zero_sorted = sorted(vals_close_zero,key=(lambda x: np.abs(x)))
            vals_renormalized = [e/vals_close_zero_sorted[0] for e in vals_close_zero]
            mask2 = vals_close_zero == vals_close_zero_sorted[0]
            if np.abs(vals_renormalized[1]) < 100:
                print("eigen values")
                print(vals[mask])
                raise ValueError("There are more than one stready states values")
            self.rho_stready_state = vec_r[mask][mask2][0]
        self.rho_stready_state = vec_r[mask][0]
        self.rho_stready_state /=\
            self.get_operator_in_spin_sector(
                (self.Lindbladian.liouville_ops.left_vacuum
                 ).transpose().conjugate(),
                sector_right=(0, 0))\
            * self.rho_stready_state

        self.left_vacuum = self.get_operator_in_spin_sector(
            (self.Lindbladian.liouville_ops.left_vacuum
             ).transpose().conjugate(),
            sector_right=(0, 0)).todense()

        # self.check_sector(self.rho_stready_state)

    def sectors_exact_decomposition(self, set_lindblad=True, sectors=None):
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
        if (self.sign != -1.0) and set_lindblad:
            self.set_lindbladian(-1.0)
            self.precalc_correlator = {n: {} for n in self.correlators.keys()}
        self.vals_sector = {}
        self.vec_l_sector = {}
        self.vec_r_sector = {}
        if sectors is None:
            sectors = self.spin_combination
        for sector in sectors:
            L_sector = self.get_operator_in_spin_sector(self.Lindbladian.L_tot,
                                                        tuple(sector),
                                                        tuple(sector))
            self.vals_sector[tuple(sector)], self.vec_l_sector[tuple(sector)],\
                self.vec_r_sector[tuple(sector)] = \
                ed_lind.exact_spectral_decomposition(L_sector.todense())

    def set_spin_sectors_fermionic_ops(self, site=None):
        if site is None:
            site = int((self.nsite-1)/2)
        self.spin_sector_fermi_ops = {'c': {}, 'c_tilde': {},
                                      'cdag': {}, 'cdag_tilde': {}}
        cdag_up_sector = {}
        cdag_up_tilde_sector = {}
        c_up_sector = {}
        c_up_tilde_sector = {}
        cdag_do_sector = {}
        cdag_do_tilde_sector = {}
        c_do_sector = {}
        c_do_tilde_sector = {}

        for sector in self.spin_combination:
            sector = tuple(sector)
            up_plus = add_sectors((1, 0), sector)
            up_minus = add_sectors((-1, 0), sector)
            do_plus = add_sectors((0, 1), sector)
            do_minus = add_sectors((0, -1), sector)

            if up_plus in self.spin_combination:
                cdag_up_sector[up_plus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.cdag(site, 'up'),
                        up_plus, sector)
                cdag_up_tilde_sector[up_plus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.cdag_tilde(site, 'up'),
                        up_plus, sector)

            if up_minus in self.spin_combination:
                c_up_sector[up_minus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.c(site, 'up'),
                        up_minus, sector)
                c_up_tilde_sector[up_minus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.c_tilde(site, 'up'),
                        up_minus, sector)

            if do_plus in self.spin_combination:
                cdag_do_sector[do_plus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.cdag(site, 'do'),
                        do_plus, sector)
                cdag_do_tilde_sector[do_plus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.cdag_tilde(site, 'do'),
                        do_plus, sector)

            if do_minus in self.spin_combination:
                c_do_sector[do_minus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.c(site, 'do'),
                        do_minus, sector)
                c_do_tilde_sector[do_minus, sector] = \
                    self.get_operator_in_spin_sector(
                        self.Lindbladian.liouville_ops.c_tilde(site, 'do'),
                        do_minus, sector)

        self.spin_sector_fermi_ops['c']['up'] = c_up_sector
        self.spin_sector_fermi_ops['c']['do'] = c_do_sector
        self.spin_sector_fermi_ops['c_tilde']['up'] = c_up_tilde_sector
        self.spin_sector_fermi_ops['c_tilde']['do'] = c_do_tilde_sector

        self.spin_sector_fermi_ops['cdag']['up'] = cdag_up_sector
        self.spin_sector_fermi_ops['cdag']['do'] = cdag_do_sector
        self.spin_sector_fermi_ops['cdag_tilde']['up'] = cdag_up_tilde_sector
        self.spin_sector_fermi_ops['cdag_tilde']['do'] = cdag_do_tilde_sector

    def get_two_point_correlator_time(self, times, B, A, sector):
        """Calculate the two point correlation function of operators A and B in
        time domain.
        For now, both have to be single fermionic operators.

        Parameters
        ----------
        times : array_like, (dim,)
            Array containing the time grid, for which the correlation
            function is calculated

        B : scipy.sparse.csc_matrix (dim, dim)
            Fermionic operator.
            It is assumed, that the relevant contrebution to the correlation
            function stems from connecting the (0,0) and the "sector" sectors.

        A : scipy.sparse.csc_matrix (dim, dim)
            Fermionic operator.
            It is assumed, that the relevant contrebution to the correlation
            function stems from connecting the (0,0) and the "sector" sectors.

        sector : tuple, (int,int)
            Sector which operator A and B connect to the sector (0,0).

        Returns
        -------
        out: tuple, (numpy.ndarray,numpy.ndarray)
            containing the correlators necessary to determine the retarded and
            keldysh single particle green's functions in the time domain.
        """
        # calculating the Lindblaian time evolution operator at time 'time'
        # Set up Permutation operators, which permute the relevant sector
        # to the upper left of the matrix to be transformed and dropping the
        # rest.
        origin_sector = (0, 0)
        minus_sector = tuple(-1 * np.array(sector))

        # calculate the Operators in the given sectors
        A_sector = self.get_operator_in_spin_sector(
            A, sector, origin_sector).todense()
        A_dagger_sector = self.get_operator_in_spin_sector(
            A.transpose().conjugate(), minus_sector, origin_sector).todense()

        B_sector = self.get_operator_in_spin_sector(
            B, origin_sector, sector).todense()

        B_dagger_sector = self.get_operator_in_spin_sector(
            B.transpose().conjugate(), origin_sector, minus_sector).todense()

        left_vacuum_00 = self.get_operator_in_spin_sector(
            (self.Lindbladian.liouville_ops.left_vacuum
             ).transpose().conjugate(),
            sector_right=origin_sector).todense()

        return _get_two_point_correlator_time(
            A_sector, A_dagger_sector, B_sector, B_dagger_sector, times,
            self.vals_sector[(*sector,)], self.vals_sector[minus_sector],
            self.vec_l_sector[(*sector,)], self.vec_l_sector[minus_sector],
            self.vec_r_sector[(*sector,)], self.vec_r_sector[minus_sector],
            left_vacuum_00, self.rho_stready_state)

    def get_two_point_correlator_frequency(self, omegas, B, A, sector):
        """Calculate the two point correlation function of operators A and B in
        frequency domain.
        For now, both have to be single fermionic operators.

        Parameters
        ----------
        times : array_like, (dim,)
            Array containing the time grid, for which the correlation
            function is calculated

        B : scipy.sparse.csc_matrix (dim, dim)
            Fermionic operator.
            It is assumed, that the relevant contrebution to the correlation
            function stems from connecting the (0,0) and the "sector" sectors.

        A : scipy.sparse.csc_matrix (dim, dim)
            Fermionic operator.
            It is assumed, that the relevant contrebution to the correlation
            function stems from connecting the (0,0) and the "sector" sectors.

        sector : tuple, (int,int)
            Sector which operator A and B connect to the sector (0,0).

        Returns
        -------
        out: tuple, (numpy.ndarray,numpy.ndarray)
            containing the correlators necessary to determine the retarded and
            keldysh single particle green's functions in the frequency domain.
        """
        # Set up Permutation operators, which permute the relevant sector
        # to the upper left of the matrix to be transformed and dropping the
        # rest.
        origin_sector = (0, 0)
        minus_sector = tuple(-1 * np.array(sector))

        # calculate the Operators in the given sectors
        A_sector = self.get_operator_in_spin_sector(
            A, sector, origin_sector).todense()
        A_dagger_sector = self.get_operator_in_spin_sector(
            A.transpose().conjugate(), minus_sector, origin_sector).todense()

        B_sector = self.get_operator_in_spin_sector(
            B, origin_sector, sector).todense()

        B_dagger_sector = self.get_operator_in_spin_sector(
            B.transpose().conjugate(), origin_sector, minus_sector).todense()

        left_vacuum_00 = self.get_operator_in_spin_sector(
            (self.Lindbladian.liouville_ops.left_vacuum
             ).transpose().conjugate(),
            sector_right=origin_sector).todense()

        return _get_two_point_correlator_frequency(
            A_sector, A_dagger_sector, B_sector, B_dagger_sector, omegas,
            self.vals_sector[(*sector,)],
            self.vals_sector[minus_sector], self.vec_l_sector[(*sector,)],
            self.vec_l_sector[minus_sector], self.vec_r_sector[(*sector,)],
            self.vec_r_sector[minus_sector],
            left_vacuum_00, self.rho_stready_state)

    def precalculate_expectation_value(self, operator_keys,shape):
        # precalculate the needed terms for calculating the corresponding
        # correlation function
        _precalculate_expectation_value(operator_keys,shape,
                                        self.precalc_correlator,
                                        self.left_vacuum,
                                        self.spin_sector_fermi_ops,
                                        self.vec_r_sector,
                                        self.vec_l_sector,
                                        self.rho_stready_state)


    def get_correlator_component_time(self, contour_operators, times):
        # times (0,...,tmax)
        # contour_operators list of (1/0,'c/cdag','do/up')
        # create for all elements incontour_operators a
        # contour_parameters set, with (1/0,-\+1 or 0,'c/cdag','do/up')
        #   -> the last element should be have t = 0
        n = len(contour_operators)
        t_max = t_max_possitions(n)
        green_shape =tuple([2*len(times)-1 for i in range(n-1)])
        # print(green_shape)
        contour_parameters_list = [[(x[0],t,x[1],x[2]) for x,t in
                               zip(contour_operators,ts)] for ts in t_max]
        green_component = np.zeros(green_shape,dtype=np.complex128)
        # assert contour_parameters == contour_ordering(contour_parameters)
        for contour_parameters in contour_parameters_list:
            # get contour parameters with the last time substracted from
            # the other contour times
            contour_parameters = contour_ordering(contour_parameters)
            contour_parameters_new, t_max_steady_state = steady_state_contour(
                contour_parameters)

            # applying the quantum regression theorem on the contour_parameters
            contour_parameters_new, t_max_steady_state = quantum_regresion_ordering(
                contour_parameters_new, t_max_steady_state)

            # generate dictionary keys for precalculated expectation values
            operator_keys = get_operator_keys(contour_parameters_new,
                                            self.operator_sectors)

            # shape of the precalculated expectation value
            tensor_shape = tuple([self.vals_sector[key[2][1]].shape[0]
                        for key in operator_keys[:-1]])

            # print(t_max_steady_state)
            # print(contour_parameters_new)
            # print(operator_keys)
            # calculate the matrix products appearing the correlators
            # without the time or frequency dependence
            self.precalculate_expectation_value(operator_keys,tensor_shape)
            # calculate the time dependent greensfunction component
            self.get_correlator_time(green_component, times, t_max_steady_state,
                            operator_keys, tensor_shape)
        return green_component


    def get_correlator_time(self,green_component, times, t_max_steady_state,
                            operator_keys, tensor_shape):
        n = len(operator_keys)
        for nt in range(len(times)):
            # calculate the time range for which the green's function has to
            # be calculated
            time_ranges = get_ranges(nt,np.array(t_max_steady_state))
            time_ranges_shape = tuple([x[1]-x[0]+1 for x in time_ranges])
            print(nt)
            # print(time_ranges)
            # print(time_ranges_shape)
            N_time_range = np.prod(time_ranges_shape)
            for n_time_range in range(N_time_range):
                n_converted = confert_to_tuple(n_time_range,time_ranges_shape)
                # print(n_converted)
                greens_time,indices = get_greens_component_times(times, time_ranges, n_converted)
                # print(greens_time,indices)

            # n = len(operator_keys)
                N_precalc_corr = np.prod(tensor_shape)
                G = 0+0j
                time_evol_sectors = list(map(lambda x: x[2][0], operator_keys))[1:]
                for n_precalc_corr in range(N_precalc_corr):
                    idx = confert_to_tuple(n_precalc_corr,tensor_shape)
                    # print(idx)
                    tmp = 1.+0.j
                    # print(idx,time_evol_sectors,times)
                    for i, sector, t in zip(idx, time_evol_sectors, greens_time):
                        # print(i,sector,t)
                        # print(np.exp(self.vals_sector[sector][i]*np.abs(t)))
                        tmp *= np.exp(self.vals_sector[sector][i]*np.abs(t))
                    # print(tmp)
                    G += tmp*self.precalc_correlator[n][operator_keys][tuple(idx)]

                green_component[tuple(indices)] = G

    def get_correlator_component_frequency(self, contour_operators, freq):
        # times (0,...,tmax)
        # contour_operators list of (1/0,'c/cdag','do/up')
        # create for all elements incontour_operators a
        # contour_parameters set, with (1/0,-\+1 or 0,'c/cdag','do/up')
        #   -> the last element should be have t = 0
        n = len(contour_operators)
        t_max = t_max_possitions(n)
        green_shape =tuple([len(freq) for i in range(n-1)])
        # print(green_shape)
        contour_parameters_list = [[(x[0],t,x[1],x[2]) for x,t in
                               zip(contour_operators,ts)] for ts in t_max]
        green_component = np.zeros(green_shape,dtype=np.complex128)
        # assert contour_parameters == contour_ordering(contour_parameters)
        for contour_parameters in contour_parameters_list:
            # get contour parameters with the last time substracted from
            # the other contour times
            contour_parameters = [(*x,i) for i,x in
                                    enumerate(contour_parameters)]
            contour_parameters = contour_ordering(contour_parameters)
            permutation_key = tuple([x[-1] for x in contour_parameters])
            sign = self.permutation_signs[n][permutation_key]
            contour_parameters = [x[:-1] for x in contour_parameters]

            contour_parameters_new, t_max_steady_state = steady_state_contour(
                contour_parameters)

            # applying the quantum regression theorem on the contour_parameters
            contour_parameters_new, t_max_steady_state = quantum_regresion_ordering(
                contour_parameters_new, t_max_steady_state)

            # generate dictionary keys for precalculated expectation values
            operator_keys = get_operator_keys(contour_parameters_new,
                                            self.operator_sectors)

            # shape of the precalculated expectation value
            tensor_shape = tuple([self.vals_sector[key[2][1]].shape[0]
                        for key in operator_keys[:-1]])

            # print(t_max_steady_state)
            # print(contour_parameters_new)
            # print(operator_keys)

            # calculate the matrix products appearing the correlators
            # without the time or frequency dependence
            self.precalculate_expectation_value(operator_keys,tensor_shape)
            # calculate the time dependent greensfunction component
            self.get_correlator_frequency(green_component, freq, t_max_steady_state,
                            operator_keys, tensor_shape, sign)
        return green_component*(-1j)**(n//2)*sign

    def get_correlator_frequency(self,green_component, freq, t_max_steady_state,
                            operator_keys, tensor_shape,sign):
        n = len(operator_keys)
        freq_shape = tuple([len(freq)for i in range(n-1)])
        N_freq = np.prod(freq_shape)
        for nw in range(N_freq):
            # calculate the time range for which the green's function has to
            # be calculated
            freq_index = confert_to_tuple(nw,freq_shape)
            greens_freq = [freq[i] for i in freq_index]

            N_tensor = np.prod(tensor_shape)
            G = 0+0j
            sectors = list(map(lambda x: x[2][0], operator_keys))[1:]
            for n_tensor in range(N_tensor):
                idx = confert_to_tuple(n_tensor,tensor_shape)
                tmp = 1.+0.j
                for i, s, w in zip(idx, sectors, greens_freq):
                    tmp *= 1./((1j*w+self.vals_sector[s][i]))

                G += tmp*self.precalc_correlator[n][operator_keys][tuple(idx)]

            green_component[tuple(freq_index)] = G

    # 1) [X]sort on contour time -> should be done before
    # 2) [X]substract the last time index from all others
    # 3) [X]check if a timedifference is negative is negative
    #       -> permute the operator to the end
    #       -> use tilde rule to move the operator from the end
    #          past the density operator
    #           -> rho c(t)c(t') = c_tilde(t')c_tilde(t)rho
    #              oder          = c_tilde(t)c_tilde(t')rho
    # 4) [x] get operators in right order
    # 5) [x] insert time propagation operator
    # 6) [ ] calculate the correlators, without time or frequency and eigenvalues
    #        [X] -> split precalculation of correlators and calculation of
    #              correlation function for a given time
    #        [ ] -> given symmetries and operators, e.g. 'c'/'cdag' and
    #              'up'/'do' generate all precalculated correlators
    #        [X] -> of given contour and times/frequency calculate the
    #               corresponding Green's function
    # 6) []  calculate the correlators with time/ frequency and eigenvalues

    def set_contour_symmetries(self, n_correlators):
        self.contour_symmetries = {}
        for n in n_correlators:
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
                        # print("combinations",comb_symmetric[t_max_idx])
                        comb_symmetric[t_max_idx] = tuple([x if i != 0
                                                           else 0 for i, x in enumerate(comb[t_max_idx])])
                        n_correlator_symmetrie[tuple(comb_symmetric)] = comb
            if n==2:
                choose_components([(1,0),(0,1)], n_correlator_symmetrie)
            self.contour_symmetries[n] = n_correlator_symmetrie.copy()

@njit(cache=True)
def get_ranges(n_tmax,list_of_times):
    ranges = np.zeros((len(list_of_times),2),dtype=np.int64)
    if len(list_of_times)==1:
        ranges[0] = [n_tmax*list_of_times[0],n_tmax*list_of_times[0]]
        return ranges

    for i,l in enumerate(list_of_times):
        if l ==0:
            ranges[i] = [-n_tmax,n_tmax]
        elif l ==-1:
            ranges[i] = [-n_tmax,0]
        elif l == 1:
            ranges[i] = [n_tmax,n_tmax]
    return ranges

@njit(cache=True)
def get_greens_component_times(real_times, time_ranges, n_converted):
    times_value = np.zeros(len(n_converted),dtype=np.float64)
    indices = np.zeros(len(n_converted),dtype=np.int64)
    for i in range(len(n_converted)):
        time_range = time_ranges[i][1]-time_ranges[i][0]
        if time_range == 0:
            j = time_ranges[i][0]
            times_value[i] = np.sign(j)*real_times[np.abs(j)]
            indices[i] = len(real_times)-1 + j
        else:
            j = time_ranges[i][0]+n_converted[i]
            times_value[i] = np.sign(j)*real_times[np.abs(j)]
            indices[i] = len(real_times)-1 + j
    return times_value,indices


# @njit(parallel=True, cache=True)
def _precalculate_expectation_value(operator_keys,shape,precalc_correlator,left_vacuum,
                                    spin_sector_fermi_ops,vec_r_sector,
                                    vec_l_sector,rho_stready_state):
    n = len(operator_keys)
    N_precalc_corr = np.prod(shape)

    precalc_corr_tmp = np.zeros(shape, dtype=np.complex128)
    # print(shape)
    if operator_keys not in precalc_correlator[n].keys():
        ecpectation_start = left_vacuum.dot(
            (spin_sector_fermi_ops[operator_keys[0][0]]
                )[operator_keys[0][1]][operator_keys[0][2]].todense())
        for n_precalc_corr in range(N_precalc_corr):
            idx = confert_to_tuple(n_precalc_corr,shape)
            # print(idx)
            expectation_val = ecpectation_start
            for i, op_key in zip(idx, operator_keys[1:]):
                # print(i,op_key[2])
                # print("before", expectation_val.shape)
                expectation_val = expectation_val.dot(
                        vec_r_sector[op_key[2][0]][i]).dot(
                        vec_l_sector[op_key[2][0]][i]).dot(
                            spin_sector_fermi_ops[op_key[0]]
                            [op_key[1]][op_key[2]].todense())
                # print("after", expectation_val.shape)
            precalc_corr_tmp[tuple(idx)] = expectation_val.dot(
                rho_stready_state)
        precalc_correlator[n][operator_keys] = precalc_corr_tmp

@njit(cache=True)
def confert_to_tuple(n,shape):
    index = np.zeros(len(shape),dtype=np.int64)
    tmp = shape[::-1]
    for i,max in enumerate(tmp):
        index[i] = n%max
        n = n//max
    return index[::-1]

@njit(parallel=True, cache=True)
def _get_two_point_correlator_time(A_sector, A_dagger_sector, B_sector,
                                   B_dagger_sector, times, vals_sector,
                                   vals_minus_sector, vec_l_sector,
                                   vec_l_minus_sector, vec_r_sector,
                                   vec_r_minus_sector,
                                   left_vacuum_00, rho_stready_state):
    """Calculate the two point correlation function of operators A and B in
        time domain. Optimized by use of numba jit.

    Parameters
    ----------
    A_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector "sector"
        to spin sector  (0,0).

    A_dagger_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector
        "minus_sector" to spin sector  (0,0).

    B_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector "sector"
        to spin sector  (0,0).

    B_dagger_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector
        "minus_sector" to spin sector  (0,0).

    times : numpy.ndarray (dim,)
        Time grid.

    vals_sector : numpy.ndarray (dim,)
        Eigenvalues of the Lindbladian in the spin sector "sector".

    vals_minus_sector : numpy.ndarray (dim,)
        Eigenvalues of the Lindbladian in the spin sector "minus_sector".

    vec_l_sector : numpy.ndarray (dim, 1, dim)
        Left eigenvectors of the Lindbladian in the spin sector "sector".

    vec_l_minus_sector : numpy.ndarray (dim, 1, dim)
        Left eigenvectors of the Lindbladian in the spin sector "minus_sector".

    vec_r_sector : numpy.ndarray (dim, dim, 1)
        Right eigenvectors of the Lindbladian in the spin sector "sector".

    vec_r_minus_sector : numpy.ndarray (dim, dim, 1)
        Right eigenvectors of the Lindbladian in the spin sector
        "minus_sector".

    left_vacuum_00 : numpy.ndarray (1, dim)
        Left vacuum vector in the spin sector (0,0).

    rho_stready_state : numpy.ndarray (dim, 1)
        Steady-state density of states in the spin sector (0,0).

    Returns
    -------
    (G_times_plus, G_times_minus): tuple (numpy.ndarray, numpy.ndarray)
        Tuple of single particle green's functions in time domain.
    """
    G_times_plus = np.zeros(times.shape, dtype=np.complex128)
    G_times_minus = np.zeros(times.shape, dtype=np.complex128)

    G_plus_tmp = np.zeros(vals_sector.shape[0], dtype=np.complex128)
    G_minus_tmp = np.zeros(vals_minus_sector.shape[0], dtype=np.complex128)
    for m in range(vals_sector.shape[0]):
        G_plus_tmp[m] = left_vacuum_00.dot(B_sector).dot(
            vec_r_sector[m]).dot(
            vec_l_sector[m]).dot(A_sector).dot(
            rho_stready_state)[0, 0]
    for m in range(vals_minus_sector.shape[0]):
        G_minus_tmp[m] = np.conj(
            left_vacuum_00.dot(B_dagger_sector).dot(
                vec_r_minus_sector[m]).dot(
                vec_l_minus_sector[m]).dot(A_dagger_sector).dot(
                rho_stready_state
            )[0, 0])

    for i, time in enumerate(times):
        for m in range(vals_sector.shape[0]):
            G_times_plus[i] += -1.j * heaviside(time, 0) \
                * np.exp(vals_sector[m] * time) * G_plus_tmp[m]

        for m in range(vals_minus_sector.shape[0]):
            G_times_minus[i] += -1.j * heaviside(time, 0) \
                * np.exp(vals_minus_sector[m] * time) * G_minus_tmp[m]
    return G_times_plus, G_times_minus


@njit(parallel=True, cache=True)
def _get_two_point_correlator_frequency(A_sector, A_dagger_sector, B_sector,
                                        B_dagger_sector, omegas, vals_sector,
                                        vals_minus_sector, vec_l_sector,
                                        vec_l_minus_sector, vec_r_sector,
                                        vec_r_minus_sector,
                                        left_vacuum_00, rho_stready_state):
    """Calculate the two point correlation function of operators A and B in
    frequency domain. Optimized by use of numba jit.

    Parameters
    ----------
    A_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector "sector"
        to spin sector  (0,0).

    A_dagger_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector
        "minus_sector" to spin sector  (0,0).

    B_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector "sector"
        to spin sector  (0,0).

    B_dagger_sector : numpy.ndarray (dim, dim)
        Fermionic operator, in subspace connecting the spin sector
        "minus_sector" to spin sector  (0,0).

    omegas : numpy.ndarray (dim,)
        Frequency grid.

    vals_sector : numpy.ndarray (dim,)
        Eigenvalues of the Lindbladian in the spin sector "sector".

    vals_minus_sector : numpy.ndarray (dim,)
        Eigenvalues of the Lindbladian in the spin sector "minus_sector".

    vec_l_sector : numpy.ndarray (dim, 1, dim)
        Left eigenvectors of the Lindbladian in the spin sector "sector".

    vec_l_minus_sector : numpy.ndarray (dim, 1, dim)
        Left eigenvectors of the Lindbladian in the spin sector "minus_sector".

    vec_r_sector : numpy.ndarray (dim, dim, 1)
        Right eigenvectors of the Lindbladian in the spin sector "sector".

    vec_r_minus_sector : numpy.ndarray (dim, dim, 1)
        Right eigenvectors of the Lindbladian in the spin sector
        "minus_sector".

    left_vacuum_00 : numpy.ndarray (1, dim)
        Left vacuum vector in the spin sector (0,0).

    rho_stready_state : numpy.ndarray (dim, 1)
        Steady-state density of states in the spin sector (0,0).

    Returns
    -------
    (G_omega_plus, G_omega_minus): tuple (numpy.ndarray, numpy.ndarray)
        Tuple of single particle green's functions in frequency domain.
    """

    G_omega_plus = np.zeros(omegas.shape, dtype=np.complex128)
    G_omega_minus = np.zeros(omegas.shape, dtype=np.complex128)

    G_plus_tmp = np.zeros(vals_sector.shape[0], dtype=np.complex128)
    G_minus_tmp = np.zeros(vals_sector.shape[0], dtype=np.complex128)
    for m in range(vals_sector.shape[0]):
        G_plus_tmp[m] = left_vacuum_00.dot(B_sector).dot(
            vec_r_sector[m]).dot(
            vec_l_sector[m]).dot(A_sector).dot(
            rho_stready_state)[0, 0]
    for m in range(vals_minus_sector.shape[0]):
        G_minus_tmp[m] = np.conj(left_vacuum_00.dot(
            B_dagger_sector).dot(
            vec_r_minus_sector[m]).dot(
            vec_l_minus_sector[m]).dot(
            A_dagger_sector).dot(rho_stready_state))[0, 0]

    for n, omega in enumerate(omegas):
        for m in range(vals_sector.shape[0]):
            G_omega_plus[n] += G_plus_tmp[m] * (
                1.0 / (omega
                       - 1j * vals_sector[m]))

        for m in range(vals_minus_sector.shape[0]):
            G_omega_minus[n] += G_minus_tmp[m] * (
                1.0 / (omega - 1j * np.conj(
                    vals_minus_sector[m])))
    return G_omega_plus, G_omega_minus

def choose_components(components, contour_symmetries):
    for key in contour_symmetries:
        value = contour_symmetries[key]
        contour_component = tuple([x[0] for x in key])
        if contour_component in components:
            if value is not None:
                contour_symmetries[value] = key
                contour_symmetries[key] = None

def get_branch_combinations(n, contour_ordered=False):
    # forward branch 0/ backward branch 1
    # Returns all possible branch combinations
    combos = []
    for i in range(2**n):
        compination = list(map(int, bin(i).replace("0b", "")))
        padding = [0 for i in range(n-len(compination))]
        combos.append(padding + compination)
    combos = np.array(combos)
    if contour_ordered:
        mask = list(map(lambda x: np.all(
            x == sorted(x, reverse=True)), combos))
        return list(map(lambda x: tuple(x), combos[mask]))
    return list(map(lambda x: tuple(x), combos))

def get_operator_keys(contour_parameters, operator_sectors):
    sectors = [(0, 0), *[operator_sectors[p[2]][p[3]]
                         for p in contour_parameters], (0, 0)]
    operators = [(p[2], p[3]) if p[1] >= 0 else (p[2]+"_tilde", p[3])
                 for p in contour_parameters]
    tmp = (0, 0)
    spin_sector_transitions = []
    for s in sectors[::-1][1:]:
        sum = add_sectors(s, tmp)
        spin_sector_transitions.append((sum, tmp))
        tmp = sum
    spin_sector_transitions = spin_sector_transitions[::-1][1:]
    operator_keys = tuple(map(lambda x: (*x[0], x[1]),
                              zip(operators, spin_sector_transitions)))
    assert spin_sector_transitions[0][0] == (0, 0), "The number of creation and "\
        + "annihilation have to be the same per spin."
    return operator_keys

def get_permutations_sign(n):
    index = list(np.ndindex(n, n))
    index_string_array = np.array(list(map(lambda x: 'a'+"".join(x),
                                           list(map(lambda y: (str(y[0]), str(y[1])), index)))))
    index_string_array = index_string_array.reshape((n, n))
    A = Matrix(index_string_array)
    A_determinante_str = A.det().__str__()

    permutation = {}
    topermute = A_determinante_str.replace("-", "+").split(" + ")
    for i, s in enumerate(topermute):
        s_tuple = tuple(s.replace('a', "").split('*'))
        sign = ""
        if i == 0:
            sign = 1
        else:
            if A_determinante_str.split(s)[0][-2] == '-':
                sign = -1
            else:
                sign = 1
        key = [None]*len(s_tuple)

        for j in s_tuple:
            key[int(j[1])] = int(j[0])

        permutation[tuple(key)] = sign
    return permutation


def contour_ordering(list_of_parameters):
    return sorted(list_of_parameters, key=(lambda x: (-x[0], x[1]) if x[0] == 1 else (-x[0], -x[1])))


def steady_state_contour(contour_parameters):
    contour_parameters_new = list(map(lambda x: (x[0],
                                                 x[1]-contour_parameters[-1][1], *x[2:]), contour_parameters))
    real_times = list(map(lambda x: x[1], contour_parameters_new))[:-1]
    return contour_parameters_new, real_times


def quantum_regresion_ordering(list_of_parameters, times):

    idx = None
    for i, x in enumerate(list_of_parameters):
        if x[1] < 0:
            idx = i

    if idx == None:
        return list_of_parameters, times
    else:
        return (list_of_parameters[(idx+1):] + list_of_parameters[:(idx+1)],
                times[:-1][(idx+1):] + times[:(idx+1)])


def add_sectors(sector1, sector2):
    return tuple([sum(x) for x in zip(sector1, sector2)])

def t_max_possitions(n):
    t_max = list(map(lambda x: tuple(x), itertools.permutations(
                [0 if i != n-1 else 1 for i in range(n)], r=n)))
    return set(t_max)

def find_index_max_time(list_of_parameters):
    return list_of_parameters.index(max(list_of_parameters, key=(lambda x: x[1])))



# TODO: I) function to calculate a given set of configuration for n-point
#          correlator.
#             1) check order is correct.
#                   -> (contour,time,creation/annihilation,spin)
#             2) check that number of spin up and spin down operator are such,
#                that we end up in (0,0) sector.
#                   -> associate +1(-1) with creator (annihilator) sort spin
#                      but tuple (up,do)
#                   -  iterate through list of tuple (see 1) ) add and substract
#                      in according spin e.g. (-1,0)
#                   XXX: What happens when I act with a tilde creation operator
#                        on rho?
#                        -> this has to be checked same change in sector as
#                           ordinary operator
#             3) Account for contour + use quantum regression theorem
#             4) If we have e.g n correlator function, than n-1 time evolution
#                operators need to be inserted
#             5) Also here we can construct a n-1-tensor for each realization
#                of a n-point correlation function, in which the expectation
#                value is precalculated. The eigenvalues are inserted
#                afterwards, with corresponding frequency or time and
#                eigenvalues
#                -> this can be done in two steps
#                   i)   first construct all permutations of keldysh contour
#                        operators
#                   ii)  calculate and store all expectation values without the
#                        explicit time or frequency dependent term
#                   iii) For realization calculate the explicit time or
#                        frequency correlator when needed
#       XXX: If the expectation values, whitout the time/frequencies are
#            saved as tensors than what
#               -> either I can calculate the whole two particle operator at ones
#                  and afterwards the four point vertex
#               -> or I can do it in one go
#       II) Function to calculate all possible correlation components
#           along the contour
#           1) construct all permutations for a given set of correlators
#           2) construct all possible combinations of contour placement
#               -> use symmetries to reduce the number of possibilities
#
# %%
if __name__ == "__main__":
    # Set parameters
    Nb = 1
    nsite = 2 * Nb + 1
    ws = np.linspace(-3, 3, 201)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.2 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])
    Us = np.zeros(nsite)
    # plt.figure()
    for U in [0]:
        Us[Nb] = U

        # Initializing auxiliary system and E, Gamma1 and Gamma2 for a
        # particle-hole symmetric system
        sys = aux.AuxiliarySystem(Nb, ws)
        sys.set_ph_symmetric_aux(es, ts, gamma)

        green = fg.FrequencyGreen(sys.ws)
        green.set_green_from_auxiliary(sys)
        hyb_aux = green.get_self_enerqy()

        # Initializing Lindblad class
        spinless = False
        L = lind.Lindbladian(nsite, spinless,
                             Dissipator=lind.Dissipator_thermal_bath)

        # Setting unitary part of Lindbladian
        T_mat = sys.E
        T_mat[Nb, Nb] = -Us[Nb] / 2.0
        L.set_unitay_part(T_mat=T_mat, U_mat=Us)

        # Setting dissipative part of Lindbladian
        L.set_dissipation(sys.Gamma1, sys.Gamma2)

        # Setting total Lindbladian
        L.set_total_linbladian()

        # Setup a correlator object
        corr = Correlators(L, 1)
        corr.update_model_parameter(sys.Gamma1, sys.Gamma2, T_mat, Us)
        corr.set_rho_steady_state()
        corr.sectors_exact_decomposition()

        #Calcolate Green's functions
        G_greater = corr.get_correlator_component_frequency(
            [(1,'c','up'),(0,'cdag','up')], ws)
        G_lesser = corr.get_correlator_component_frequency(
            [(0,'c','up'),(1,'cdag','up')], ws)

        G_R = G_lesser - G_greater
        G_K = np.conj(G_lesser)- G_greater - np.conj(np.conj(G_lesser)-G_greater)

        green2 = fg.FrequencyGreen(sys.ws, retarded=G_R, keldysh=G_K)
        sigma = green2.get_self_enerqy() - hyb_aux

        # Visualize results

        # plt.plot(sys.ws, green.retarded.imag)
        plt.figure()
        plt.plot(sys.ws, G_R.imag)
        plt.ylabel(r"$A(\omega)$")
        plt.xlabel(r"$\omega$")
        plt.show()
        plt.figure()
        plt.plot(sys.ws, G_K.imag)
        plt.ylabel(r"$A(\omega)$")
        plt.xlabel(r"$\omega$")
        plt.show()
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
