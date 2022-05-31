# %%
from typing import Union
import numpy as np
from scipy import sparse
import src.hilber_space.define_fock_space_operators as fop


class SuperFermionicOperators:
    """SuperFermionicOperators(nsite: int, spinless: bool = False,
    tilde_conjugationrule_phase: bool = True)

    Class of fermionic operators in the super-fermionic space,
    constructed form fermionic operators in Fock space.
    The fermions can have a 1/2 spin or be spinless.

    This class follows the publication by Dzhioev et. al ( J. Chem. Phys.
    134, 044121 (2011); https://doi.org/10.1063/1.3548065).

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

    Delta_N_up: scipy.sparse.csc_matrix (dim, dim)
        Difference of total paricel number of electrons with spin up
        between "normal" and "tilde" space

    Delta_N_do: scipy.sparse.csc_matrix (dim, dim)
        Difference of total paricel number of electrons with spin down
        between "normal" and "tilde" space

    N: scipy.sparse.csc_matrix (dim, dim)
        Total paricel number of in "normal" space

    N_tilde: scipy.sparse.csc_matrix (dim, dim)
        Total paricel number of in "tilde" space

    unity_tilde: scipy.sparse.csc_matrix (dim, dim)
        Matrix encoding commutation relation between "normal" and "tilde"
        space.

    tilde_operator_name: dict
        Dictionary linking the fermionic operators in the "normal" space to
        the fermionic operators of the "tilde" space.

    tilde_operator_sign: dict
        Dictionary containing the phase picked up by transitioning
        fermionic operators from "normal" to "tilde" space.

    """

    def __init__(self, nsite: int, spinless: bool = False,
                 tilde_conjugationrule_phase: bool = True) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.fock_ops = fop.FermionicFockOperators(
            nsite, spinless, sorted_particle_number=False)
        self.tilde_conjugationrule_phase = tilde_conjugationrule_phase

        # Define Spin sector operators N_{\simga}-\tilde{N}_{\simga}
        if not spinless:
            self.Delta_N_up = self.get_super_fermionic_operator(
                self.fock_ops.N_up) \
                - self.get_super_fermionic_tilde_operator(self.fock_ops.N_up)
            self.Delta_N_do = self.get_super_fermionic_operator(
                self.fock_ops.N_do) \
                - self.get_super_fermionic_tilde_operator(self.fock_ops.N_do)

        self.N = self.get_super_fermionic_operator(self.fock_ops.N)
        self.N_tilde = self.get_super_fermionic_tilde_operator(self.fock_ops.N)

        # Set up operator encoding commutation relation between
        self.unity_tilde = sparse.csc_matrix(
            ([1.0], ([0], [0])), shape=(1, 1))
        for _ in range(self.fock_ops.spin_times_site):
            self.unity_tilde = sparse.kron(self.fock_ops.sigma_z,
                                           self.unity_tilde, format="csc")

        # Set up left vacuum state |I> with or witout the complex phase in the
        # tilde conjugation rule.
        self.left_vacuum = sparse.lil_matrix(
            (4**(self.fock_ops.spin_times_site), 1), dtype=np.complex64)

        # Find and Set |0>x|0>
        vacuum_index = np.where(
            (self.N + self.N_tilde).diagonal() == 0)  # |0>x|0>
        assert len(vacuum_index) == 1
        self.left_vacuum[vacuum_index[0][0], 0] = 1

        # Set up the constructing operator for the left vacuum state
        sign = 1
        if self.tilde_conjugationrule_phase:
            sign = -1j
        left_vacuum_constructor = sparse.eye(
            4**(self.fock_ops.spin_times_site), dtype=np.complex64)
        if not spinless:
            for site in range(self.fock_ops.nsite):
                for spin in ["up", "do"]:
                    left_vacuum_constructor = left_vacuum_constructor * (
                        sparse.eye(4**(self.fock_ops.spin_times_site),
                                   dtype=np.complex64) + sign * self.cdag(
                                       site, spin) * self.cdag_tilde(
                                           site, spin))
        else:
            for site in range(self.fock_ops.nsite):
                left_vacuum_constructor = left_vacuum_constructor * (
                    sparse.eye(4**(self.fock_ops.spin_times_site),
                               dtype=np.complex64) + sign * self.cdag(
                                   site) * self.cdag_tilde(site))
        # construct the left vacuum state
        self.left_vacuum = left_vacuum_constructor * self.left_vacuum

        # Dictionary for changing to the tilde space operators
        self.tilde_operator_name = {'c': 'cdag_tilde', 'cdag': 'c_tilde'}
        if self.tilde_conjugationrule_phase:
            self.tilde_operator_sign = {'c': 1, 'cdag': 1, 'cdag_tilde': -1j,
                                        'c_tilde': -1j}
        else:
            self.tilde_operator_sign = {
                'c': 1, 'cdag': 1, 'cdag_tilde': 1, 'c_tilde': -1}

    def c(self, ii: int, spin: Union[str, None] = None) -> sparse.csc_matrix:
        """Returns the super-fermionic space annihilation operator of site
        'ii' and spin 'spin'.

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
            super-fermionic space annihilation operator of site 'ii' and
            spin 'spin'.
        """
        return sparse.kron(self.fock_ops.c(ii, spin),
                           self.unity_tilde,
                           format="csc")

    def cdag(self, ii: int, spin: Union[str, None] = None
             ) -> sparse.csc_matrix:
        """Returns the super-fermionic space creation operator of site 'ii' and
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
            super-fermionic space creation operator of site 'ii' and
            spin 'spin'.
        """
        return sparse.kron(self.fock_ops.cdag(ii, spin),
                           self.unity_tilde,
                           format="csc")

    def c_tilde(self, ii: int, spin: Union[str, None] = None) -> sparse.csc_matrix:
        """Returns the super-fermionic space tilde annihilation operator of site 'ii'
        and spin 'spin'.

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
            super-fermionic space tilde annihilation operator of site 'ii' and
            spin 'spin'.
        """

        return sparse.kron(sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=np.complex64, format="csc"),
                           self.fock_ops.c(ii, spin),
                           format="csc")

    def cdag_tilde(self, ii: int, spin: Union[str, None] = None) -> sparse.csc_matrix:
        """Returns the super-fermionic space tilde creation operator of site 'ii' and
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
            super-fermionic space tilde creation operator of site 'ii' and
            spin 'spin'.
        """

        return sparse.kron(sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=np.complex64, format="csc"),
                           self.fock_ops.cdag(ii, spin),
                           format="csc")

    def n_channel(self, ii: int, channel: str = 'ch') -> sparse.csc_matrix:
        """Returns the super-fermionic 'normal' space charge or spin density
        operator at site 'ii'.

        Parameters
        ----------
        ii : int
            Site or orbital index

        channel : string, optional
            Channel index 'ch','x', 'y' or 'z', by default 'ch'.

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            super-fermionic space charge or spin density operator at site 'ii'.
        """
        return self.get_super_fermionic_operator(self.fock_ops.n_channel(
            ii=ii, channel=channel))

    def n_channel_tilde(self, ii: int, channel: str = 'ch') -> sparse.csc_matrix:
        """Returns the super-fermionic 'tilde' space charge or spin density
        operator at site 'ii'.

        Parameters
        ----------
        ii : int
            Site or orbital index

        channel : string, optional
            Channel index 'ch','x', 'y' or 'z', by default 'ch'.

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            super-fermionic space charge or spin density operator at site 'ii'.
        """
        return self.get_super_fermionic_tilde_operator(self.fock_ops.n_channel(
            ii=ii, channel=channel))

    def get_super_fermionic_operator(self, fock_operator: sparse.csc_matrix
                                     ) -> sparse.csc_matrix:
        """Returns the super-fermionic space representation of an Fock space operator

        These super-fermionic space operators are used, when they are appear
        left from the density matrix in the Fock space formulation of, e.g the
        Lindblad equation.

        Parameters
        ----------
        fock_operator : scipy.sparse.csc_matrix (dim, dim)
            Operator in Fock space representation


        Returns
        -------
        out: scipy.sparse.csc_matrix (dim**2, dim**2)
            super-fermionic space operators acting from the left
        """
        assert fock_operator.shape == (
            2**self.fock_ops.spin_times_site,
            2**self.fock_ops.spin_times_site)
        return sparse.kron(fock_operator,
                           sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=np.complex64, format="csc"),
                           format="csc")

    def get_super_fermionic_tilde_operator(self,
                                           fock_operator: sparse.csc_matrix
                                           ) -> sparse.csc_matrix:
        """Returns the super-fermionic space tilde representation of an Fock
        space operator

        These super-fermionic space operators are used, when they are appear
        right from the density matrix in the Fock space formulation of, e.g the
        Lindblad equation.

        Parameters
        ----------
        fock_operator : scipy.sparse.csc_matrix (dim, dim)
            Operator in Fock space representation


        Returns
        -------
        out: scipy.sparse.csc_matrix (dim**2, dim**2)
            super-fermionic space tilde operators acting.
        """
        assert fock_operator.shape == (
            2**self.fock_ops.spin_times_site,
            2**self.fock_ops.spin_times_site)

        return sparse.kron(sparse.eye(2**self.fock_ops.spin_times_site,
                                      dtype=np.complex64, format="csc"),
                           fock_operator.transpose(), format="csc")


# %%
if __name__ == "__main__":
    nsite = 2
    fl_op = SuperFermionicOperators(nsite, tilde_conjugationrule_phase=True)

    print("Checking commutation relation")

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=np.complex64, format="csc")

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

                    # mixed creation and annihilation operators
                    anti_commutation_mixed_c_cdag_tilde = \
                        fop.anti_commutator(fl_op.c(i, s1),
                                            fl_op.cdag_tilde(j, s2))
                    anti_commutation_mixed_c_c_tilde = \
                        fop.anti_commutator(fl_op.c(i, s1),
                                            fl_op.c_tilde(j, s2))
                    anti_commutation_mixed_cdag_cdag_tilde = \
                        fop.anti_commutator(
                            fl_op.cdag(i, s1), fl_op.cdag_tilde(j, s2))

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
                        assert anti_commutation_fock_c_c.nnz == 0
                        assert anti_commutation_fock_cdag_cdag.nnz == 0

                        assert (anti_commutation_tilde_c_cdag -
                                identity_super_fermionic).nnz == 0
                        assert anti_commutation_tilde_c_c.nnz == 0
                        assert anti_commutation_tilde_cdag_cdag.nnz == 0

                    else:
                        assert anti_commutation_fock_c_cdag.nnz == 0
                        assert anti_commutation_fock_c_c.nnz == 0
                        assert anti_commutation_fock_cdag_cdag.nnz == 0

                        assert anti_commutation_mixed_c_cdag_tilde.nnz == 0
                        assert anti_commutation_mixed_c_c_tilde.nnz == 0
                        assert anti_commutation_mixed_cdag_cdag_tilde.nnz == 0

                        assert anti_commutation_tilde_c_cdag.nnz == 0
                        assert anti_commutation_tilde_c_c.nnz == 0
                        assert anti_commutation_tilde_cdag_cdag.nnz == 0

    print("Commutation relations fulfilled")

    print("Checking the tilde conjugation rule")
    n_nonzeros = 0
    if fl_op.tilde_conjugationrule_phase:
        for i in range(nsite):
            for spin in ["up", "do"]:
                n_nonzeros += (fl_op.c(i, spin) + 1j * fl_op.cdag_tilde(
                    i, spin)).dot(fl_op.left_vacuum).nnz
                n_nonzeros += (fl_op.cdag(i, spin) + 1j * fl_op.c_tilde(
                    i, spin)).dot(fl_op.left_vacuum).nnz
    else:
        for i in range(nsite):
            for spin in ["up", "do"]:
                n_nonzeros += (fl_op.c(i, spin) - fl_op.cdag_tilde(i, spin)
                               ).dot(fl_op.left_vacuum).nnz
                n_nonzeros += (fl_op.cdag(i, spin) + fl_op.c_tilde(i, spin)
                               ).dot(fl_op.left_vacuum).nnz
    if n_nonzeros != 0:
        raise ValueError("ERROR: super-fermionic creation/annihilation"
                         + " operator and tilde super-fermionic "
                         + "creation/annihilation operator acting on the"
                         + " left vacuum state have to be same.")
    else:
        print("super-fermionic creation/annihilation operator and"
              + " tilde super-fermionic creation/annihilation operator "
              + "acting on the left vacuum state are the same.")

    N = sparse.kron(fl_op.fock_ops.N,
                    sparse.eye(2**fl_op.fock_ops.spin_times_site,
                               dtype=np.complex64, format="csc"))

    N_T = sparse.kron(sparse.eye(2**fl_op.fock_ops.spin_times_site,
                                 dtype=np.complex64, format="csc"),
                      fl_op.fock_ops.N)
    operator_equal_operator_tilse = N.dot(fl_op.left_vacuum) - \
        N_T.dot(fl_op.left_vacuum)
    if operator_equal_operator_tilse.count_nonzero() == 0:
        print("Operator acting in Fock supspace and transposed " +
              "acting in tilde supspace in superfermion results in same" +
              " super-fermionic vector.")
    else:
        raise ValueError("ERROR: Operator in fock subspace in " +
                         "superfermion representation acting on left vacuum " +
                         "vector has to be equal to transposed operator in " +
                         "tilde subspace acting on left vacuum vector")
# %%
