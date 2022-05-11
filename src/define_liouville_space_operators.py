# %%
import numpy as np
from scipy import sparse
import src.define_fock_space_operators as fop

# XXX: For now the super-fermionic representation is not used with the complex
#      phase as surguessted by Dzhioev et. al ( J. Chem. Phys. 134, 044121
#      (2011); https://doi.org/10.1063/1.3548065). This is not necessary, since
#      we are using the Schroedinger picture:
#      Tr[A(t,0)\rho] = Tr[V^{\dagger}(t,0)A\rho] = Tr[AV(t,0)\rho]
#      where V(t,0) describes the time-evolution of the density

class FermionicLouvilleOperators:
    def __init__(self, nsite, spinless=False, tilde_conjugationrule_phase=False):
        """Class of fermionic operators in the Liouville space, constructed
        form fermionic operators in Fock space.

        The fermions can have a 1/2 spin or be spinless.

        Parameters
        ----------
        nsite : int
            number of sites/ orbitals of the fermionic system

        spinless : bool, optional
            Indicates if the fermions are spinless, by default False

        tilde_conjugationrule_phase: bool, optional
            If True a the tilde conjugation rule is applied with an imaginary
            phase for the tilde operators and the left vacuum state. If False,
            the phase is set to one, by default False


        Attributes
        ----------
        fock_ops: define_fock_space_operators.FermionicFockOperators
            Fock space object containing the creation and annihilation
            operators of a fermionic system with nsite sites/orbitals and
            spin 1/2 if spinless is None, spinless otherwise.

        left_vacuum: scipy.sparse.csc_matrix (dim,1)
            left vacuum state according to Dzhioev et. al.

        tilde_conjugationrule_phase: bool
            If True a the tilde conjugation rule is applied with an imaginary
            phase for the tilde operators and the left vacuum state. If False,
            the phase is set to one.

        transformation_tilde: scipy.sparse.csc_matrix (dim,dim)
            unitary transformation in the tilde subspace in order to account
            for the complex phase in the left vacuum state.

        Delta_N_up: scipy.sparse.csc_matrix (dim, dim)
            Difference of total paricel number of electrons with spin up
            between "normal" and "tilde" space

        Delta_N_do: scipy.sparse.csc_matrix (dim, dim)
            Difference of total paricel number of electrons with spin down
            between "normal" and "tilde" space
        """
        self.fock_ops = fop.FermionicFockOperators(nsite, spinless)
        self.tilde_conjugationrule_phase = tilde_conjugationrule_phase
        self.left_vacuum = sparse.lil_matrix(
            (4**(self.fock_ops.spin_times_site), 1), dtype=complex)

        pnums = self.fock_ops.N.diagonal()
        for n in range(2**self.fock_ops.spin_times_site):
            pnum_vector = sparse.lil_matrix(
                (2**self.fock_ops.spin_times_site, 1), dtype=complex)
            pnum_vector[n, 0] = 1.0

            if self.tilde_conjugationrule_phase:
                self.left_vacuum += ((-1j)**pnums[n]) * sparse.kron(
                    pnum_vector, pnum_vector)
            else:
                self.left_vacuum += sparse.kron(
                    pnum_vector, pnum_vector)
        self.left_vacuum = self.left_vacuum.tocsc()

        if self.tilde_conjugationrule_phase:
            transformation_tilde = sparse.lil_matrix(
                (2**self.fock_ops.spin_times_site,
                2**self.fock_ops.spin_times_site),
                dtype=complex)
            transformation_tilde.setdiag(
                (-1j)**self.fock_ops.N.diagonal())

            transformation_tilde = transformation_tilde.tocsc()

            self.transformation_tilde = sparse.kron(transformation_tilde,
                           sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=complex, format="csc"),
                           format="csc")
        if not spinless:
            self.Delta_N_up = self.get_louville_operator(self.fock_ops.N_up) \
                - self.get_louville_tilde_operator(self.fock_ops.N_up)
            self.Delta_N_do = self.get_louville_operator(self.fock_ops.N_do) \
                - self.get_louville_tilde_operator(self.fock_ops.N_do)

    def particle_number_subspace_projector(self, nelec):
        """Projector for given particle number nelec in the liouville space.


        Parameters
        ----------
        nelec : int
            Particel number.

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Projector for the desired particle number subspace.
        """
        pnum_index = np.where(self.fock_ops.N.diagonal() == nelec)[0]
        pnum_projector = sparse.lil_matrix(
            (4**(self.fock_ops.spin_times_site),
             4**(self.fock_ops.spin_times_site)), dtype=complex)
        for n in pnum_index:
            n_vector = sparse.csc_matrix(
                (2**self.fock_ops.spin_times_site, 1), dtype=complex)
            n_vector[n, 0] = 1.0
            for m in pnum_index:
                m_vector = sparse.csc_matrix(
                    (2**self.fock_ops.spin_times_site, 1), dtype=complex)
                m_vector[m, 0] = 1
                nm_vector = sparse.kron(n_vector, m_vector)
                pnum_projector += nm_vector * nm_vector.transpose()
                pnum_projector = pnum_projector.tocsc()

        return pnum_projector

    def spin_sector_projector(self, sector):
        """Projector for given spin sector "sector" in the liouville space.


        Parameters
        ----------
        sector : tuple (up, do)
            Spin sector defined by the difference between particles in
            "normal" and "tilde" space for spin up and down

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Projector for the desired Spin sector subspace, with given
            spin sector difference between "normal" and "tilde" space of
            'sector'
        """
        if not self.fock_ops.spinless:
            pnum_up_index = np.where(
                self.Delta_N_up.diagonal() == sector[0])[0]
            pnum_do_index = np.where(
                self.Delta_N_do.diagonal() == sector[1])[0]
            mask_section = np.in1d(pnum_up_index, pnum_do_index)
            pnum_index = pnum_up_index[mask_section]
            pnum_per_spin_projector = sparse.lil_matrix(
                (4**(self.fock_ops.spin_times_site),
                 4**(self.fock_ops.spin_times_site)), dtype=complex)

            for n in pnum_index:
                pnum_per_spin_projector[n, n] = 1.0
            pnum_per_spin_projector = pnum_per_spin_projector.tocsc()

            return pnum_per_spin_projector

    def spin_sector_permutation_operator(self, sector, full=False):
        """Returns a permutation operator, permuting desired spin sector
        "sector" to the upper left corner of a the liouville space matrix.

        This can be used to reduce the Lindbladian to the relevant spin
        sectors. And accelerating calculations such as the exact
        diagonalization and time propagation.

        Parameters
        ----------
        sector : tuple (up, do)
            Spin sector defined by the difference between particles in
            "normal" and "tilde" space for spin up and down

        full : bool, optional
            If False it returns the permutation operator that permutes the
            relevant spin sector to the upper left of the matrix and projects
            out the rest. If True the full permutation operator, which doesn't
            project out the rest is returned. By default False

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Permutation operator for the desired Spin sector subspace, with
            given spin sector difference between "normal" and "tilde" space of
            'sector'
        """
        if not self.fock_ops.spinless:
            pnum_up_index = np.where(
                self.Delta_N_up.diagonal() == sector[0])[0]
            pnum_do_index = np.where(
                self.Delta_N_do.diagonal() == sector[1])[0]
            mask_section = np.in1d(pnum_up_index, pnum_do_index)
            pnum_index = np.sort(pnum_up_index[mask_section])
            dim_subspace = pnum_index.shape[0]
            total_index = np.linspace(0.,
                                      4**(self.fock_ops.spin_times_site) - 1.,
                                      num=4**(self.fock_ops.spin_times_site),
                                      dtype=int)
            n_prime = total_index[dim_subspace:]
            m_prime = np.setdiff1d(total_index, pnum_index)
            mask_n_m_prime_same = np.in1d(n_prime, m_prime)
            n_m_prime_same = n_prime[mask_n_m_prime_same]
            n_prime_diff = np.setdiff1d(n_prime, m_prime)
            m_prime_diff = np.setdiff1d(m_prime, n_prime)

            perm_op_sector = sparse.lil_matrix(
                (4**(self.fock_ops.spin_times_site),
                 4**(self.fock_ops.spin_times_site)), dtype=complex)

            for n in range(dim_subspace):
                perm_op_sector[n, pnum_index[n]] = 1.0

            if full:
                for n, m in zip(n_prime_diff, m_prime_diff):
                    perm_op_sector[n, m] = 1.0

                for n in n_m_prime_same:
                    perm_op_sector[n, n] = 1.0

            perm_op_sector = perm_op_sector.tocsc()
            perm_op_sector = perm_op_sector
            return dim_subspace, perm_op_sector

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
            If the system is spinless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space annihilation operator of site 'ii' and
            spin 'spin'.
        """
        return sparse.kron(self.fock_ops.c(ii, spin),
                           sparse.eye(2**self.fock_ops.spin_times_site,
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
            If the system is spinless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space creation operator of site 'ii' and
            spin 'spin'.
        """
        return sparse.kron(self.fock_ops.cdag(ii, spin),
                           sparse.eye(2**self.fock_ops.spin_times_site,
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
            If the system is spinless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space tilde annihilation operator of site 'ii' and
            spin 'spin'.
        """

        return sparse.kron(sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=complex, format="csc"),
                           self.fock_ops.c(ii, spin),
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
            If the system is spinless, than the string is optional

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Liouville space tilde creation operator of site 'ii' and
            spin 'spin'.
        """

        return sparse.kron(sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=complex, format="csc"),
                           self.fock_ops.cdag(ii, spin),
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
            2**self.fock_ops.spin_times_site,
            2**self.fock_ops.spin_times_site)
        return sparse.kron(fock_operator,
                           sparse.eye(2**self.fock_ops.spin_times_site,
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
            2**self.fock_ops.spin_times_site,
            2**self.fock_ops.spin_times_site)

        return sparse.kron(sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=complex, format="csc"),
                           fock_operator.transpose(), format="csc")


# %%
if __name__ == "__main__":
    nsite = 1
    fl_op = FermionicLouvilleOperators(nsite,tilde_conjugationrule_phase=True)

    print("Checking commutation relation")

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
                    anti_commutation_fock_c_cdag = \
                        fop.anti_commutator(fl_op.c(i, s1), fl_op.cdag(j, s2))
                    anti_commutation_fock_c_c = \
                        fop.anti_commutator(fl_op.c(i, s1), fl_op.c(j, s2))

                    anti_commutation_fock_cdag_cdag = \
                        fop.anti_commutator(
                            fl_op.cdag(i, s1), fl_op.cdag(j, s2))

                    # purely tilde creation and annihilation operators
                    anti_commutation_tilde_c_cdag = \
                        fop.anti_commutator(
                            fl_op.c_tilde(i, s1), fl_op.cdag_tilde(j, s2))

                    anti_commutation_tilde_c_c =  \
                        fop.anti_commutator(
                            fl_op.c_tilde(i, s1), fl_op.c_tilde(j, s2))
                    anti_commutation_tilde_cdag_cdag = fop.anti_commutator(
                        fl_op.cdag_tilde(i, s1), fl_op.cdag_tilde(j, s2))

                    if i == j and s1 == s2:
                        assert (anti_commutation_fock_c_cdag -
                                identity_super_fermionic).nnz == 0
                        assert (
                            anti_commutation_tilde_c_cdag-
                                identity_super_fermionic).nnz == 0

                        assert anti_commutation_fock_c_c.nnz == 0
                        assert anti_commutation_tilde_c_c.nnz == 0

                        assert anti_commutation_fock_cdag_cdag.nnz == 0
                        assert anti_commutation_tilde_cdag_cdag.nnz == 0

                    else:
                        assert anti_commutation_fock_c_cdag.nnz == 0
                        assert anti_commutation_tilde_c_cdag.nnz == 0

                        assert anti_commutation_fock_c_c.nnz == 0
                        assert anti_commutation_tilde_c_c.nnz == 0

                        assert anti_commutation_fock_cdag_cdag.nnz == 0
                        assert anti_commutation_tilde_cdag_cdag.nnz == 0

    print("Commutation relations fulfilled")

    print("Checking the tilde conjugation rule")
    n_nonzeros = 0
    if fl_op.tilde_conjugationrule_phase:
        for i in range(nsite):
            for spin in ["up", "do"]:
                n_nonzeros += (fl_op.c(i, spin) + 1j * fl_op.cdag_tilde(i, spin)).dot(
                                fl_op.left_vacuum).nnz
                n_nonzeros += (fl_op.cdag(i, spin) - 1j * fl_op.c_tilde(i, spin)).dot(
                                fl_op.left_vacuum).nnz
    else:
        for i in range(nsite):
            for spin in ["up", "do"]:
                n_nonzeros += (fl_op.c(i, spin) - fl_op.cdag_tilde(i, spin)).dot(
                                fl_op.left_vacuum).nnz
                n_nonzeros += (fl_op.cdag(i, spin) - fl_op.c_tilde(i, spin)).dot(
                                fl_op.left_vacuum).nnz
    if n_nonzeros != 0:
        raise ValueError("ERROR: Liouville creation/annihilation operator and"
                         + " tilde Liouville creation/annihilation operator "
                         + "acting on the left vacuum state have to be same.")
    else:
        print("Liouville creation/annihilation operator and"
              + " tilde Liouville creation/annihilation operator "
              + "acting on the left vacuum state are the same.")

    N = sparse.kron(fl_op.fock_ops.N,
                    sparse.eye(4**fl_op.fock_ops.nsite,
                               dtype=complex, format="csc"))

    N_T = sparse.kron(sparse.eye(4**fl_op.fock_ops.nsite,
                                 dtype=complex, format="csc"),
                      fl_op.fock_ops.N)
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
