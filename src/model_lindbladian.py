# %%
import numpy as np
from scipy import sparse, linalg
import src.define_liouville_space_operators as lop
import src.model_hamiltonian as ham


def Dissipator_thermal_bath(Gamma1, Gamma2, liouville_operators):
    """Retruns the dissipator of a fermionic system coupled to a thermal
    fermionic bath, therefore the particle number can change due to the
    dissipator.


    Parameters
    ----------
    Gamma1 : numpy.ndarray (dim,dim)
        2D array with the coupling to thermal bath, describing the removal of
        electrons

    Gamma2 : numpy.ndarray (dim,dim)
        2D array with the coupling to thermal bath, describing the injection of
        electrons

    liouville_operators : FermionicLouvilleOperators
        Contains the fermionic creation and annihilation operators Liouville
        space and converts Fock space operators in to Liouville space operators

    Returns
    -------
    (L_Gamma1,L_Gamma2): scipy.sparse.csc_matrix (dim,dim)
        Tuple of the Liouville dissipator, with L_Gamma1, describing the
        removal of electrons, and L_Gamma2, describing the injection of
        electrons.
    """
    nsite = liouville_operators.fock_operators.nsite
    spin_times_site = liouville_operators.fock_operators.spin_times_site
    L_Gamma1 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    L_Gamma2 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    if liouville_operators.fock_operators.spinless:
        spins = [None]
    else:
        spins = ["up", "do"]

    for ii in range(nsite):
        for jj in range(nsite):
            for spin in spins:
                if Gamma1[ii, jj] != 0:
                    L_Gamma1 += (2. * Gamma1[ii, jj]
                                 * liouville_operators.c(jj, spin)
                                 * liouville_operators.cdag_tilde(ii, spin)
                                 - Gamma1[ii, jj]
                                 * (liouville_operators.cdag(ii, spin)
                                    * liouville_operators.c(jj, spin)
                                    + liouville_operators.c_tilde(jj, spin)
                                    * liouville_operators.cdag_tilde(ii, spin))
                                 )
                if Gamma2[ii, jj] != 0:
                    L_Gamma2 += (2. * Gamma2[ii, jj]
                                 * liouville_operators.cdag(ii, spin)
                                 * liouville_operators.c_tilde(jj, spin)
                                 - Gamma2[ii, jj]
                                 * (liouville_operators.c(jj, spin)
                                    * liouville_operators.cdag(ii, spin)
                                    + liouville_operators.cdag_tilde(ii, spin)
                                    * liouville_operators.c_tilde(jj, spin))
                                 )
    return L_Gamma1, L_Gamma2


def Dissipator_thermal_radiation_mode(Gamma1, Gamma2, liouville_operators):
    """Retruns the dissipator of a fermionic system coupled to a single mode
    bosonic bath.

    Parameters
    ----------
    Gamma1 : numpy.ndarray (dim,dim)
        2D array with the coupling to thermal bath, describing the removal of
        electrons

    Gamma2 : numpy.ndarray (dim,dim)
        2D array with the coupling to thermal bath, describing the injection of
        electrons

    liouville_operators : FermionicLouvilleOperators
        Contains the fermionic creation and annihilation operators Liouville
        space and converts Fock space operators in to Liouville space operators

    Returns
    -------
    (L_Gamma1,L_Gamma2): scipy.sparse.csc_matrix (dim,dim)
        Tuple of the Liouville dissipator, with L_Gamma1, describing the
        removal of electrons, and L_Gamma2, describing the injection of
        electrons.
    """
    nsite = liouville_operators.fock_operators.nsite
    spin_times_site = liouville_operators.fock_operators.spin_times_site
    L_Gamma1 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    L_Gamma2 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    if liouville_operators.fock_operators.spinless:
        spins = [None]
    else:
        spins = ["up", "do"]

    for ii in range(nsite):
        for jj in range(nsite):
            for spin in spins:
                if jj > ii:
                    if Gamma1[ii, jj] != 0:
                        L_Gamma1 += (Gamma1[ii, jj]
                                     * liouville_operators.cdag(ii, spin)
                                     * liouville_operators.c(jj, spin)
                                     * liouville_operators.c_tilde(ii, spin)
                                     * liouville_operators.cdag_tilde(jj, spin)
                                     - 0.5 * Gamma1[ii, jj]
                                     * (liouville_operators.cdag(jj, spin)
                                        * liouville_operators.c(ii, spin)
                                        * liouville_operators.cdag(ii, spin)
                                        * liouville_operators.c(jj, spin)
                                        + liouville_operators.c_tilde(jj, spin)
                                        * liouville_operators.cdag_tilde(ii,
                                        spin)
                                        * liouville_operators.c_tilde(ii, spin)
                                        * liouville_operators.cdag_tilde(jj,
                                        spin))
                                     )

                    if Gamma2[ii, jj] != 0:
                        L_Gamma2 += (Gamma2[ii, jj]
                                     * liouville_operators.cdag(jj, spin)
                                     * liouville_operators.c(ii, spin)
                                     * liouville_operators.c_tilde(jj, spin)
                                     * liouville_operators.cdag_tilde(ii, spin)
                                     - 0.5 * Gamma2[ii, jj]
                                     * (liouville_operators.cdag(ii, spin)
                                        * liouville_operators.c(jj, spin)
                                        * liouville_operators.cdag(jj, spin)
                                        * liouville_operators.c(ii, spin)
                                        + liouville_operators.c_tilde(ii, spin)
                                        * liouville_operators.cdag_tilde(jj,
                                        spin)
                                        * liouville_operators.c_tilde(jj, spin)
                                        * liouville_operators.cdag_tilde(ii,
                                        spin))
                                     )
    return L_Gamma1, L_Gamma2


class Lindbladian:
    def __init__(self, nsite, spinless=False, unitary_transformation=False,
                 Hamiltonian=ham.hubbard_hamiltonian,
                 Dissipator=Dissipator_thermal_bath) -> None:
        """Class for setting up a Lindbladian

        Parameters
        ----------
        nsite : int
            number of sites/ orbitals of the fermionic system

        spinless : bool, optional
            Indicates if the fermions are spinless, by default False

        unitary_transformation : bool, optional
            If True a unitary transformation on the subspace of the tilde
            operators and the left vacuum state is preformed, by default False

        Hamiltonian : function returning a sparse matrix, optional
            Hamiltonian of the reduced system, by default
            ham.hubbard_hamiltonian

        Dissipator : function returning a tuple of sparse matrices, optional
            Function describing the dissipation/coupling between the reduced
            system and the surrounding bath, by default Dissipator_thermal_bath

        Attributes
        ----------
        liouville_operators:  FermionicLouvilleOperators
        Contains the fermionic creation and annihilation operators Liouville
        space and converts Fock space operators in to Liouville space operators

        Hamiltonian: function returning a sparse matrix, optional
            Hamiltonian of the reduced system, by default
            ham.hubbard_hamiltonian

        Dissipator: function returning a tuple of sparse matrices, optional
            Function describing the dissipation/coupling between the reduced
            system and the surrounding bath, by default Dissipator_thermal_bath

        L_tot: scipy.sparse.csc_matrix (dim,dim)
            Full Lindbladian sparse matrix

        L_unitary: scipy.sparse.csc_matrix (dim,dim)
            Unitary component of the Lindbladian sparse matrix

        L_Gamma1: scipy.sparse.csc_matrix (dim,dim)
            Dissipative component of the Lindbladian sparse matrix, describing
            the removal of electrons

        L_Gamma2: scipy.sparse.csc_matrix (dim,dim)
            Dissipative component of the Lindbladian sparse matrix,
            describing the injection of electrons

        vals: numpy.ndarray (dim,)
            Containing the eigen values of the Lindbladian

        vec_l: numpy.ndarray (dim, 1, dim)
            Contains the left eigen vectors ot the Lindbladian. The order
            corresponds to the order of the eigen vectors

        vec_r: numpy.ndarray (dim, dim, 1)
            Contains the right eigen vectors ot the Lindbladian. The order
            corresponds to the order of the eigen vectors
        """
        self.liouville_operators = lop.FermionicLouvilleOperators(
            nsite,
            spinless=spinless,
            unitary_transformation=unitary_transformation)
        self.Hamiltonian = Hamiltonian
        self.Dissipator = Dissipator
        self.L_tot = sparse.csc_matrix(
            (4**(self.liouville_operators.fock_operators.spin_times_site),
             4**(self.liouville_operators.fock_operators.spin_times_site)),
            dtype=complex)

    def set_unitay_part(self, T_mat, U_mat):
        """Set the unitary part of the Lindbladian describing a unitary
        propagation

        Parameters
        ----------
        T_mat : numpy.ndarry (dim, dim)
            Hopping matrix of the Hamiltonian

        U_mat : numpy.ndarry (dim, dim)
            Two particle interaction matrix of the Hamiltonian
        """
        Hamil_Fock = self.Hamiltonian(
            T_mat, U_mat,
            eop=self.liouville_operators.fock_operators)
        self.L_unitary = -1.j * \
            (self.liouville_operators.get_louville_operator(Hamil_Fock)
             - self.liouville_operators.get_louville_tilde_operator(Hamil_Fock)
             )

    def set_dissipation(self, Gamma1, Gamma2):
        """Set the dissipative part of the Lindbladian describing a non-unitary
        propagation

        Parameters
        ----------
        Gamma1 : numpy.ndarray (dim,dim)
            2D array with the coupling to thermal bath, describing the removal
            of electrons

        Gamma2 : numpy.ndarray (dim,dim)
            2D array with the coupling to thermal bath, describing the
            injection of electrons

        Raises
        ------
        ValueError
            If Gamma matrices have mismatching shapes
        """
        nsite = self.liouville_operators.fock_operators.nsite
        if (Gamma1.shape != Gamma2.shape) and Gamma1.shape != (nsite, nsite):
            raise ValueError("ERROR: Wrong shape of Gamma matrices. They have"
                             + F" to be {(nsite,nsite)}.")

        self.L_Gamma1, self.L_Gamma2 = self.Dissipator(
            Gamma1, Gamma2, self.liouville_operators)

    def set_total_linbladian(self):
        """Set the total Lindbladian"""
        self.L_tot = self.L_unitary + self.L_Gamma1 + \
            self.L_Gamma2

    def exact_spectral_decomposition(self):
        """Exact diagonalization of the Lindbladian, setting the attributes
        vals, containing the eigen values, vec_l and vec_r containing the left
        and right eigen vectors respectively.
        """
        n_spin_site = self.liouville_operators.fock_operators.spin_times_site
        self.vals, vec_r = linalg.eig(
            self.L_tot.toarray())
        for i in range(4**(n_spin_site)):
            for j in range(4**(n_spin_site)):
                vec_r[j, i] = round(vec_r[j, i].real, 14) + \
                    1.j * round(vec_r[j, i].imag, 14)

        vec_l_ = np.linalg.inv(vec_r.T)
        self.vec_l = []
        self.vec_r = []

        for i in range(4**(n_spin_site)):
            self.vec_r.append(vec_r[:, i].reshape((4**(n_spin_site), 1)))
            self.vec_l.append(vec_l_[:, i].reshape((1, 4**(n_spin_site))))

        self.vec_l = np.array(self.vec_l)
        self.vec_r = np.array(self.vec_r)

    def time_propagation_all_times_exact_diagonalization(self, times, vec0):
        """Retruns the time propagated Liouvillian vectors vec for all times
        "times", with vec0 as starting vector.

        Parameters
        ----------
        times : numpy.ndarray (dim,)
            array of times for which the propagated vector is to be calculated

        vec0 : numpy.ndarray (dim,1)
            Initial Liouvillian vector.

        Returns
        -------
        vec : numpy.ndarray (dim, dim2)
            Time propagated Liouvillian vectors. vec[:,i] contains the time
            propagated vector at time times[i]
        """
        dim = vec0.shape[0]
        vec = np.zeros((dim,) + times.shape, dtype=complex)
        for l1 in np.arange(dim):
            vec += np.exp(self.vals[l1] * times)[None, :] * self.vec_r[l1] * (
                (self.vec_l[l1]).dot(vec0))
        return vec

    def time_propagation_exact_diagonalization(self, time, vec0):
        """Retruns the time propagated Liouvillian vector vec, with vec0 as
        starting vector.

        Parameters
        ----------
        times : numpy.ndarray (dim,)
            array of times for which the propagated vector is to be calculated

        vec0 : numpy.ndarray (dim,1)
            Initial Liouvillian vector.

        Returns
        -------
        vec : numpy.ndarray (dim, dim2)
            Time propagated Liouvillian vectors. vec[:,i] contains the time
            propagated vector at time times[i]
        """
        dim = vec0.shape[0]
        vec = np.zeros((dim, 1), dtype=complex)
        for l1 in np.arange(dim):
            vec += np.exp(self.vals[l1] * time) * self.vec_r[l1] * (
                (self.vec_l[l1]).dot(vec0))
        return vec
