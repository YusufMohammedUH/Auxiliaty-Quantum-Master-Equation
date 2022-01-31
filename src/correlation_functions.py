"""Here the base code for the construction of correlation functions
    will be written.
    """
import numpy as np
from numba import njit
import src.model_lindbladian as lind
import src.lindbladian_exact_decomposition as ed_lind
import src.auxiliary_system_parameter as aux
import src.frequency_greens_function as fg
from src.dos_util import heaviside
import matplotlib.pyplot as plt


class Correlators:
    def __init__(self, Lindbladian, spin_sector_max):
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
        self.spin_sector_max = spin_sector_max
        self.set_spin_sectors()

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
        n_steady_state = vals[mask].shape[0]
        if n_steady_state > 1:
            raise ValueError("There are more than one stready states")
        self.rho_stready_state = vec_r[mask][0]
        self.rho_stready_state /=\
            self.get_operator_in_spin_sector(
                (self.Lindbladian.liouville_ops.left_vacuum
                 ).transpose().conjugate(),
                sector_right=(0, 0))\
            * self.rho_stready_state

        # self.check_sector(self.rho_stready_state)

    def sectors_exact_decomposition(self, set_lindblad=True):
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
        self.vals_sector = {}
        self.vec_l_sector = {}
        self.vec_r_sector = {}
        for sector in self.spin_combination:
            L_sector = self.get_operator_in_spin_sector(self.Lindbladian.L_tot,
                                                        tuple(sector),
                                                        tuple(sector))
            self.vals_sector[tuple(sector)], self.vec_l_sector[tuple(sector)],\
                self.vec_r_sector[tuple(sector)] = \
                ed_lind.exact_spectral_decomposition(L_sector.todense())

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
    for i, time in enumerate(times):
        time_evolution_sector = \
            ed_lind.time_evolution_operator(
                time, vals_sector,
                vec_l_sector,
                vec_r_sector)
        time_evolution_minus_sector = \
            ed_lind.time_evolution_operator(
                time, vals_minus_sector,
                vec_l_minus_sector,
                vec_r_minus_sector)

        G_times_plus[i] = -1j * heaviside(time, 0) \
            * left_vacuum_00.dot(B_sector).dot(
                time_evolution_sector).dot(A_sector).dot(
                    rho_stready_state)[0, 0]
        # TODO: check formula
        G_times_minus[i] = -1j * heaviside(time, 0) \
            * np.conj(
                left_vacuum_00.dot(B_dagger_sector).dot(
                    time_evolution_minus_sector).dot(A_dagger_sector).dot(
                    rho_stready_state
                )[0, 0])
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

    for n, omega in enumerate(omegas):
        for m in range(vals_sector.shape[0]):
            G_omega_plus[n] += left_vacuum_00.dot(B_sector).dot(
                vec_r_sector[m]).dot(
                vec_l_sector[m]).dot(A_sector).dot(
                rho_stready_state)[0, 0] * (
                1.0 / (omega
                       - 1j * vals_sector[m]))
        # TODO: check formula
        for m in range(vals_minus_sector.shape[0]):
            G_omega_minus[n] += np.conj(left_vacuum_00.dot(
                B_dagger_sector).dot(
                vec_r_minus_sector[m]).dot(
                vec_l_minus_sector[m]).dot(
                    A_dagger_sector).dot(rho_stready_state))[0, 0] * (
                        1.0 / (omega - 1j * np.conj(
                            vals_minus_sector[m])))
    return G_omega_plus, G_omega_minus


if __name__ == "__main__":
    # Set parameters
    Nb = 1
    nsite = 2 * Nb + 1
    ws = np.linspace(-5, 5, 201)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.2 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])
    Us = np.zeros(nsite)
    Us[Nb] = 2.

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

    # Calcolate Green's functions
    G_plus, G_minus = corr.get_two_point_correlator_frequency(
        ws, corr.Lindbladian.liouville_ops.c(Nb, "up"),
        corr.Lindbladian.liouville_ops.cdag(Nb, "up"), (1, 0))

    G_R = G_plus + G_minus
    G_K = G_plus + np.conj(G_minus) - np.conj(G_plus + np.conj(G_minus))
    green2 = fg.FrequencyGreen(sys.ws, retarded=G_R, keldysh=G_K)
    sigma = green2.get_self_enerqy() - hyb_aux

    # Visualize results
    plt.figure()
    plt.plot(sys.ws, green.retarded.imag)
    plt.plot(sys.ws, G_R.imag)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$ImG^R_{aux,0}(\omega)$",
                r"$ImG^R_{aux}(\omega)$"])
    plt.show()

    plt.figure()
    plt.plot(sys.ws, green.keldysh.imag)
    plt.plot(sys.ws, G_K.imag)
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
