# %%
import numpy as np
from scipy import sparse
import src.define_fock_space_operators as fop

# XXX: For now the super-fermionic representation is used with the complex
#      phase as surguessted by Dzhioev et. al ( J. Chem. Phys. 134, 044121
#      (2011); https://doi.org/10.1063/1.3548065). but i don't see any
#      advantage at the level of constructing the Lindbladian. It should be
#      checked if there is an advantage for the quantum regression theorem,
#      e.g. the one particle Green's function. If not, get rid of it.


class FermionicLouvilleOperators:
    def __init__(self, nsite, spinless=False, unitary_transformation=False):
        """Class of fermionic operators in the Liouville space, constructed
        form fermionic operators in Fock space.

        The fermions can have a 1/2 spin or be spinless.

        Parameters
        ----------
        nsite : int
            number of sites/ orbitals of the fermionic system

        spinless : bool, optional
            Indicates if the fermions are spinless, by default False

        unitary_transformation: bool, optional
            If True a unitary transformation on the subspace of the tilde
            operators and the left vacuum state is preformed, by default False

        Attributes
        ----------
        fock_operators: define_fock_space_operators.FermionicFockOperators
            Fock space object containing the creation and annihilation
            operators of a fermionic system with nsite sites/orbitals and
            spin 1/2 if spinnless is None, spinnless otherwise.

        left_vacuum: scipy.sparse.csc_matrix (dim,1)
            left vacuum state according to Dzhioev et. al.

        transformation_tilde: scipy.sparse.csc_matrix (dim,dim)
            unitary transformation in the tilde subspace in order to account
            for the complex phase in the left vacuum state.
        """
        self.fock_operators = fop.FermionicFockOperators(nsite, spinless)
        self.unitary_transformation = unitary_transformation
        self.left_vacuum = sparse.lil_matrix(
            (4**(self.fock_operators.spin_times_site), 1), dtype=complex)

        pnums = self.fock_operators.N.diagonal()
        for n in range(2**self.fock_operators.spin_times_site):
            pnum_vector = sparse.lil_matrix(
                (2**self.fock_operators.spin_times_site, 1), dtype=complex)
            pnum_vector[n, 0] = 1.0

            if self.unitary_transformation:
                self.left_vacuum += ((-1j)**pnums[n]) * sparse.kron(
                    pnum_vector, pnum_vector)
            else:
                self.left_vacuum += sparse.kron(
                    pnum_vector, pnum_vector)
        self.left_vacuum = self.left_vacuum.tocsc()
        self.transformation_tilde = sparse.lil_matrix(
            (2**self.fock_operators.spin_times_site,
             2**self.fock_operators.spin_times_site),
            dtype=complex)
        self.transformation_tilde.setdiag(
            (-1j)**self.fock_operators.N.diagonal())
        self.transformation_tilde = self.transformation_tilde.tocsc()

    def particle_number_subspace_projector(self, nelec):
        pnum_index = np.where(self.fock_operators.N.diagonal() == nelec)[0]
        pnum_projector = sparse.csc_matrix(
            (4**(self.fock_operators.spin_times_site),
             4**(self.fock_operators.spin_times_site)), dtype=complex)
        for n in pnum_index:
            n_vector = sparse.csc_matrix(
                (2**self.fock_operators.spin_times_site, 1), dtype=complex)
            n_vector[n, 0] = 1.0
            for m in pnum_index:
                m_vector = sparse.csc_matrix(
                    (2**self.fock_operators.spin_times_site, 1), dtype=complex)
                m_vector[m, 0] = 1
                nm_vector = sparse.kron(n_vector, m_vector)
                pnum_projector += nm_vector * nm_vector.transpose()
        if self.unitary_transformation:
            transform = sparse.kron(
                sparse.eye(2**self.fock_operators.spin_times_site,
                           dtype=complex, format="csc"),
                self.transformation_tilde,
                format="csc")
            return transform * pnum_projector *\
                transform.transpose().conjugate()
        else:
            return pnum_projector

    def c(self, ii, spin=None):
        """Returns the Liouville space annihilation operator of site 'ii' and
        spin 'spin'.

        These operators are used, when they are placed left from the density
        matrix in the Fock space formulation of, e.g the Lindblad equation.

        Parameters
        ----------
        ii : int
            Site or orbital index
        spin : 'up' or 'do', optional
            Spin index, by default None. For spin 1/2 fermions a string has
            to be supplied. 'do' for down spin and 'up' for spin up.
            If the system is spinnless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space annihilation operator of site 'ii' and
            spin 'spin'.
        """
        return sparse.kron(self.fock_operators.c(ii, spin),
                           sparse.eye(2**self.fock_operators.spin_times_site,
                                      dtype=complex, format="csc"),
                           format="csc")

    def cdag(self, ii, spin=None):
        """Returns the Liouville space creation operator of site 'ii' and
        spin 'spin'.

        These operators are used, when they are placed left from the density
        matrix in the Fock space formulation of, e.g the Lindblad equation.

        Parameters
        ----------
        ii : int
            Site or orbital index
        spin : 'up' or 'do', optional
            Spin index, by default None. For spin 1/2 fermions a string has
            to be supplied. 'do' for down spin and 'up' for spin up.
            If the system is spinnless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space creation operator of site 'ii' and
            spin 'spin'.
        """
        return sparse.kron(self.fock_operators.cdag(ii, spin),
                           sparse.eye(2**self.fock_operators.spin_times_site,
                                      dtype=complex, format="csc"),
                           format="csc")

    def c_tilde(self, ii, spin=None):
        """Returns the Liouville space tilde annihilation operator of site 'ii' and
        spin 'spin'.

        These operators are used, when they are placed right from the density
        matrix in the Fock space formulation of, e.g the Lindblad equation.

        Parameters
        ----------
        ii : int
            Site or orbital index
        spin : 'up' or 'do', optional
            Spin index, by default None. For spin 1/2 fermions a string has
            to be supplied. 'do' for down spin and 'up' for spin up.
            If the system is spinnless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space tilde annihilation operator of site 'ii' and
            spin 'spin'.
        """
        if self.unitary_transformation:
            return sparse.kron(
                sparse.eye(2**self.fock_operators.spin_times_site,
                           dtype=complex, format="csc"),
                self.transformation_tilde * self.fock_operators.c(
                    ii, spin).transpose()
                * self.transformation_tilde.transpose().conjugate(),
                format="csc")

        return sparse.kron(sparse.eye(2**self.fock_operators.spin_times_site,
                                      dtype=complex, format="csc"),
                           self.fock_operators.c(ii, spin).transpose(),
                           format="csc")

    def cdag_tilde(self, ii, spin=None):
        """Returns the Liouville space tilde creation operator of site 'ii' and
        spin 'spin'.

        These operators are used, when they are placed right from the density
        matrix in the Fock space formulation of, e.g the Lindblad equation.

        Parameters
        ----------
        ii : int
            Site or orbital index
        spin : 'up' or 'do', optional
            Spin index, by default None. For spin 1/2 fermions a string has
            to be supplied. 'do' for down spin and 'up' for spin up.
            If the system is spinnless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space tilde creation operator of site 'ii' and
            spin 'spin'.
        """
        if self.unitary_transformation:
            return sparse.kron(
                sparse.eye(2**self.fock_operators.spin_times_site,
                           dtype=complex, format="csc"),
                self.transformation_tilde
                * self.fock_operators.cdag(
                    ii, spin).transpose()
                * self.transformation_tilde.transpose().conjugate(),
                format="csc")

        return sparse.kron(sparse.eye(2**self.fock_operators.spin_times_site,
                                      dtype=complex, format="csc"),
                           self.fock_operators.cdag(ii, spin).transpose(),
                           format="csc")

    def get_louville_operator(self, fock_operator):
        """Returns the Liouville space representation of an Fock space operator

        These Liouville space operators are used, when they are appear left
        from the density matrix in the Fock space formulation of, e.g the
        Lindblad equation.

        Parameters
        ----------
        fock_operator : scipy.sparse.csc_matrix (dim, dim)
            Operator in Fock space representation


        Returns
        -------
        out: scipy.sparse.csc_matrix (dim**2, dim**2)
            Liouville space operators acting from the left
        """
        assert fock_operator.shape == (
            2**self.fock_operators.spin_times_site,
            2**self.fock_operators.spin_times_site)
        return sparse.kron(fock_operator,
                           sparse.eye(2**self.fock_operators.spin_times_site,
                                      dtype=complex, format="csc"),
                           format="csc")

    def get_louville_tilde_operator(self, fock_operator):
        """Returns the Liouville space tilde representation of an Fock space
        operator

        These Liouville space operators are used, when they are appear right
        from the density matrix in the Fock space formulation of, e.g the
        Lindblad equation.

        Parameters
        ----------
        fock_operator : scipy.sparse.csc_matrix (dim, dim)
            Operator in Fock space representation


        Returns
        -------
        out: scipy.sparse.csc_matrix (dim**2, dim**2)
            Liouville space tilde operators acting.
        """
        assert fock_operator.shape == (
            2**self.fock_operators.spin_times_site,
            2**self.fock_operators.spin_times_site)
        if self.unitary_transformation:
            return sparse.kron(
                sparse.eye(2**self.fock_operators.spin_times_site,
                           dtype=complex, format="csc"),
                self.transformation_tilde
                * fock_operator.transpose()
                * self.transformation_tilde.transpose().conjugate(),
                format="csc")
        return sparse.kron(sparse.eye(2**self.fock_operators.spin_times_site,
                                      dtype=complex, format="csc"),
                           fock_operator.transpose(), format="csc")


# %%
if __name__ == "__main__":
    nsite = 1
    fl_op = FermionicLouvilleOperators(nsite)

    print("Checking commutation relation")
    backtransform = sparse.kron(
        sparse.eye(4**nsite, dtype=complex, format="csc"),
        fl_op.transformation_tilde.transpose().conjugate())
    fock_projector = sparse.kron(sparse.eye(4**nsite, dtype=complex,
                                            format="csc"),
                                 sparse.csc_matrix((4**nsite, 4**nsite)))
    tilde_projector = sparse.kron(sparse.csc_matrix((4**nsite, 4**nsite)),
                                  sparse.eye(4**nsite, dtype=complex,
                                             format="csc"))
    identity_super_fermionic = sparse.eye(
        4**(2 * nsite), dtype=complex, format="csc")

    for i in range(nsite):
        for j in range(nsite):
            for s1 in ["up", "do"]:
                for s2 in ["up", "do"]:
                    # purely fock creation and annihilation operators
                    anti_commutation_fock_c_cdag = backtransform * \
                        fop.anti_commutator(fl_op.c(i, s1), fl_op.cdag(j, s2))
                    anti_commutation_fock_c_c = backtransform * \
                        fop.anti_commutator(fl_op.c(i, s1), fl_op.c(j, s2))

                    anti_commutation_fock_cdag_cdag = backtransform *\
                        fop.anti_commutator(
                            fl_op.cdag(i, s1), fl_op.cdag(j, s2))

                    # purely tilde creation and annihilation operators
                    anti_commutation_tilde_c_cdag = backtransform *\
                        fop.anti_commutator(
                            fl_op.c_tilde(i, s1), fl_op.cdag_tilde(j, s2))

                    anti_commutation_tilde_c_c = backtransform * \
                        fop.anti_commutator(
                            fl_op.c_tilde(i, s1), fl_op.c_tilde(j, s2))
                    anti_commutation_tilde_cdag_cdag = fop.anti_commutator(
                        fl_op.cdag_tilde(i, s1), fl_op.cdag_tilde(j, s2))

                    if i == j and s1 == s2:
                        assert (anti_commutation_fock_c_cdag -
                                backtransform).dot(
                                    fl_op.left_vacuum).count_nonzero() == 0
                        assert (anti_commutation_tilde_c_cdag -
                                backtransform).dot(
                                    fl_op.left_vacuum).count_nonzero() == 0

                    else:
                        assert (
                            anti_commutation_fock_c_cdag).dot(
                                fl_op.left_vacuum).count_nonzero() == 0
                        assert (
                            anti_commutation_tilde_c_cdag).dot(
                                fl_op.left_vacuum).count_nonzero() == 0

                        assert (anti_commutation_fock_c_c).dot(
                            fl_op.left_vacuum).count_nonzero() == 0
                        assert (anti_commutation_tilde_c_c).dot(
                            fl_op.left_vacuum).count_nonzero() == 0

                        assert (
                            anti_commutation_fock_cdag_cdag).dot(
                                fl_op.left_vacuum).count_nonzero() == 0
                        assert (
                            anti_commutation_tilde_cdag_cdag).dot(
                                fl_op.left_vacuum).count_nonzero() == 0

    print("Commutation relations fulfilled")

    print("Checking the tilde conjugation rule")
    n_nonzeros = 0
    for i in range(nsite):
        for spin in ["up", "do"]:
            n_nonzeros += ((fl_op.c(i, spin)
                           - fl_op.c_tilde(i, spin)).dot(
                               fl_op.left_vacuum)).count_nonzero()
            n_nonzeros += ((fl_op.cdag(i, spin)
                           - fl_op.cdag_tilde(i, spin)).dot(
                               fl_op.left_vacuum)).count_nonzero()
    if n_nonzeros != 0:
        raise ValueError("ERROR: Liouville creation/annihilation operator and"
                         + " tilde Liouville creation/annihilation operator "
                         + "acting on the left vacuum state have to be same.")
    else:
        print("Liouville creation/annihilation operator and"
              + " tilde Liouville creation/annihilation operator "
              + "acting on the left vacuum state are the same.")

    N = sparse.kron(fl_op.fock_operators.N,
                    sparse.eye(4**fl_op.fock_operators.nsite,
                               dtype=complex, format="csc"))

    N_T = sparse.kron(sparse.eye(4**fl_op.fock_operators.nsite,
                                 dtype=complex, format="csc"),
                      fl_op.fock_operators.N)
    operator_equal_operator_tilse = N.dot(fl_op.left_vacuum) - \
        N_T.dot(fl_op.left_vacuum)
    if operator_equal_operator_tilse.count_nonzero() == 0:
        print("Operator acting in Fock supspace and transposed " +
              "acting in tilde supspace in superfermion results in same" +
              " Liouville vector.")
    else:
        raise ValueError("ERROR: Operator in fock subspace in " +
                         "superfermion representation acting on left vacuum " +
                         "vector has to be equal to transposed operator in " +
                         "tilde subspace acting on left vacuum vector")
# %%
