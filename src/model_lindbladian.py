# %%
from scipy import sparse
import src.define_liouville_space_operators as lop
import src.model_hamiltonian as ham


def Dissipator_thermal_bath(Gamma1, Gamma2, liouville_ops, sign=1):
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

    liouville_ops : FermionicLouvilleOperators
        Contains the fermionic creation and annihilation operators Liouville
        space and converts Fock space operators in to Liouville space operators

    Returns
    -------
    (L_Gamma1,L_Gamma2): scipy.sparse.csc_matrix (dim,dim)
        Tuple of the Liouville dissipator, with L_Gamma1, describing the
        removal of electrons, and L_Gamma2, describing the injection of
        electrons.
    """
    nsite = liouville_ops.fock_ops.nsite
    spin_times_site = liouville_ops.fock_ops.spin_times_site
    L_Gamma1 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    L_Gamma2 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    if liouville_ops.fock_ops.spinless:
        spins = [None]
    else:
        spins = ["up", "do"]

    for ii in range(nsite):
        for jj in range(nsite):
            for spin in spins:
                if Gamma1[ii, jj] != 0:
                    L_Gamma1 += (2. * sign * Gamma1[ii, jj]
                                 * liouville_ops.c(jj, spin)
                                 * liouville_ops.cdag_tilde(ii, spin)
                                 - Gamma1[ii, jj]
                                 * (liouville_ops.cdag(ii, spin)
                                    * liouville_ops.c(jj, spin)
                                    + liouville_ops.c_tilde(jj, spin)
                                    * liouville_ops.cdag_tilde(ii, spin))
                                 )
                if Gamma2[ii, jj] != 0:
                    L_Gamma2 += (2. * sign * Gamma2[ii, jj]
                                 * liouville_ops.cdag(ii, spin)
                                 * liouville_ops.c_tilde(jj, spin)
                                 - Gamma2[ii, jj]
                                 * (liouville_ops.c(jj, spin)
                                    * liouville_ops.cdag(ii, spin)
                                    + liouville_ops.cdag_tilde(ii, spin)
                                    * liouville_ops.c_tilde(jj, spin))
                                 )
    return L_Gamma1, L_Gamma2


def Dissipator_thermal_radiation_mode(Gamma1, Gamma2, liouville_ops):
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

    liouville_ops : FermionicLouvilleOperators
        Contains the fermionic creation and annihilation operators Liouville
        space and converts Fock space operators in to Liouville space operators

    Returns
    -------
    (L_Gamma1,L_Gamma2): scipy.sparse.csc_matrix (dim,dim)
        Tuple of the Liouville dissipator, with L_Gamma1, describing the
        removal of electrons, and L_Gamma2, describing the injection of
        electrons.
    """
    nsite = liouville_ops.fock_ops.nsite
    spin_times_site = liouville_ops.fock_ops.spin_times_site
    L_Gamma1 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    L_Gamma2 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    if liouville_ops.fock_ops.spinless:
        spins = [None]
    else:
        spins = ["up", "do"]

    for ii in range(nsite):
        for jj in range(nsite):
            for spin in spins:
                if jj > ii:
                    if Gamma1[ii, jj] != 0:
                        L_Gamma1 += (Gamma1[ii, jj]
                                     * liouville_ops.cdag(ii, spin)
                                     * liouville_ops.c(jj, spin)
                                     * liouville_ops.c_tilde(ii, spin)
                                     * liouville_ops.cdag_tilde(jj, spin)
                                     - 0.5 * Gamma1[ii, jj]
                                     * (liouville_ops.cdag(jj, spin)
                                        * liouville_ops.c(ii, spin)
                                        * liouville_ops.cdag(ii, spin)
                                        * liouville_ops.c(jj, spin)
                                        + liouville_ops.c_tilde(jj, spin)
                                        * liouville_ops.cdag_tilde(ii,
                                        spin)
                                        * liouville_ops.c_tilde(ii, spin)
                                        * liouville_ops.cdag_tilde(jj,
                                        spin))
                                     )

                    if Gamma2[ii, jj] != 0:
                        L_Gamma2 += (Gamma2[ii, jj]
                                     * liouville_ops.cdag(jj, spin)
                                     * liouville_ops.c(ii, spin)
                                     * liouville_ops.c_tilde(jj, spin)
                                     * liouville_ops.cdag_tilde(ii, spin)
                                     - 0.5 * Gamma2[ii, jj]
                                     * (liouville_ops.cdag(ii, spin)
                                        * liouville_ops.c(jj, spin)
                                        * liouville_ops.cdag(jj, spin)
                                        * liouville_ops.c(ii, spin)
                                        + liouville_ops.c_tilde(ii, spin)
                                        * liouville_ops.cdag_tilde(jj,
                                        spin)
                                        * liouville_ops.c_tilde(jj, spin)
                                        * liouville_ops.cdag_tilde(ii,
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
        liouville_ops:  FermionicLouvilleOperators
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
        self.liouville_ops = lop.FermionicLouvilleOperators(
            nsite,
            spinless=spinless,
            unitary_transformation=unitary_transformation)
        self.Hamiltonian = Hamiltonian
        self.Dissipator = Dissipator
        self.L_tot = sparse.csc_matrix(
            (4**(self.liouville_ops.fock_ops.spin_times_site),
             4**(self.liouville_ops.fock_ops.spin_times_site)),
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
            eop=self.liouville_ops.fock_ops)
        self.L_unitary = -1.j * \
            (self.liouville_ops.get_louville_operator(Hamil_Fock)
             - self.liouville_ops.get_louville_tilde_operator(Hamil_Fock)
             )

    def set_dissipation(self, Gamma1, Gamma2, sign=None):
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
        nsite = self.liouville_ops.fock_ops.nsite
        if (Gamma1.shape != Gamma2.shape) and Gamma1.shape != (nsite, nsite):
            raise ValueError("ERROR: Wrong shape of Gamma matrices. They have"
                             + F" to be {(nsite,nsite)}.")
        if sign is None:
            self.L_Gamma1, self.L_Gamma2 = self.Dissipator(
                Gamma1, Gamma2, self.liouville_ops)
        else:
            self.L_Gamma1, self.L_Gamma2 = self.Dissipator(
                Gamma1, Gamma2, self.liouville_ops, sign)

    def set_total_linbladian(self):
        """Set the total Lindbladian"""
        self.L_tot = self.L_unitary + self.L_Gamma1 + \
            self.L_Gamma2
