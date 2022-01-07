# %%
import numpy as np
from scipy.special import binom
from scipy import sparse


def commutator(a, b):
    """Returns the commutation of a and b.


    Parameters
    ----------
    a : matrix_like
        Matrix like object.
    b : matrix_like
        Matrix like object.

    Returns
    -------
    out: matrix_like
        Returns a.dot(b) - b.dot(a)
    """
    return a.dot(b) - b.dot(a)


def anti_commutator(a, b):
    """Returns the anti-commutation of a and b.

    Parameters
    ----------
    a : matrix_like
        Matrix like object.
    b : matrix_like
        Matrix like object.

    Returns
    -------
    out: matrix_like
        Returns a.dot(b) + b.dot(a)
    """
    return a.dot(b) + b.dot(a)


class FermionicFockOperators:
    def __init__(self, nsite, spinless=False):
        """Class of fermiononic creation and annihilation operators in Fock
        space.

        The Operators are constructed using the Jordan-Wigner transformation
        and afterwards sorted by particle number. The fermions can be either
        spinless or can have spin 1/2.

        Parameters
        ----------
        nsite : int
            Number of sites/ orbitals in the Fermionic problem

        spinless : bool, optional
            Indicates if fermions are spinless or spin 1/2.
            If set to True, than the fermions are spinless, by default False.

        Attributes
        ----------
        spinless: bool
            Indicates if fermions are spinless or spin 1/2.

        nsite: int
            Number of sites/ orbitals in the Fermionic problem

        sigma_plus: scipy.sparse.csc_matrix (2,2)
            Pauli sigma plus matrix

        sigma_minus: scipy.sparse.csc_matrix (2,2)
            Pauli sigma minus matrix

        sigma_z: scipy.sparse.csc_matrix (2,2)
            Pauli sigma z matrix

        unit2: scipy.sparse.csc_matrix (2,2)
            Unitary matrix of dimension (2,2)

        spin_times_site: int
            Number of spins multiplied by site, necessary to determen the
            Fock space dimension

        annihilators: np.ndarray(self.spin_times_site,)
            Contains the annihilation operators. In case of spin 1/2 fermions,
            for site/orbital i the array contains the annihilator for spin up
            at 2*i and spin down at 2*i+1. In case of spinless fermions the
            annihilator of site/orbital i is given by the i-th element.

        creators: np.ndarray(self.spin_times_site,)
            Contains the creator operators. In case of spin 1/2 fermions,
            for site/orbital i the array contains the creator for spin up
            at 2*i and spin down at 2*i+1. In case of spinless fermions the
            creator of site/orbital i is given by the i-th element.

        N: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Total particle number operator.

        P: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
           Permutation operator used to sort the operators by particle number.
        TODO: Dought the need of storing P as attribute.

        pascal_indices:  list
            List of supspace dimensions with certain particle numbers.

        N_up: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Total particle operator of spin up fermions.

        N_do: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Total particle operator of spin down fermions.

        N_up_do: np.ndarray(2**self.spin_times_site,)
            list of tuples containing the diagonal of the spin up and down
            total particle operator.
        """
        if spinless:
            print("Constructing spinless fermionic Fock space.")
        self.spinless = spinless
        self.nsite = nsite
        # Jordan-Wigner transformation
        # fermionic operators

        # Set up sigma matrices
        sigma_plus = np.zeros((2, 2))
        sigma_plus[0, 1] = 1.0
        self.sigma_plus = sparse.csc_matrix(sigma_plus)

        sigma_minus = np.zeros((2, 2))
        sigma_minus[1, 0] = 1.0
        self.sigma_minus = sparse.csc_matrix(sigma_minus)

        sigma_z = np.identity(2)
        sigma_z[1, 1] = -1.0
        self.sigma_z = sparse.csc_matrix(sigma_z)

        self.unit2 = sparse.eye(2).tocsc()

        self.spin_times_site = 2 * self.nsite  # number of spins times site
        if spinless:
            self.spin_times_site = self.nsite

        # Constructing creation and annihilation operators
        self.annihilators = np.full((self.spin_times_site,), sparse.csc_matrix(
            (2**self.spin_times_site, 2**self.spin_times_site)))
        self.creators = np.full((self.spin_times_site,), sparse.csc_matrix(
            (2**self.spin_times_site, 2**self.spin_times_site)))

        for ii in range(self.spin_times_site):
            c_temp = sparse.csc_matrix(
                ([1.0], ([0], [0])), shape=(1, 1))
            cdag_temp = sparse.csc_matrix(
                ([1.0], ([0], [0])), shape=(1, 1))

            for jj in range(self.spin_times_site):
                if (jj == ii):
                    c_temp = sparse.kron(self.sigma_plus, c_temp, format="csc")
                    cdag_temp = sparse.kron(
                        self.sigma_minus, cdag_temp, format="csc")
                if (jj > ii):
                    c_temp = sparse.kron(self.unit2, c_temp, format="csc")
                    cdag_temp = sparse.kron(
                        self.unit2, cdag_temp, format="csc")
                if (jj < ii):
                    c_temp = sparse.kron(self.sigma_z, c_temp, format="csc")
                    cdag_temp = sparse.kron(
                        self.sigma_z, cdag_temp, format="csc")

            self.annihilators[ii] = c_temp
            self.creators[ii] = cdag_temp

        # Bringing operators in block-diagonal regariding the particle number
        self.N = np.sum(self.creators * self.annihilators)
        self.P = sparse.lil_matrix(
            (2**self.spin_times_site, 2**self.spin_times_site))

        pnum_diag_sorted = np.argsort(self.N.diagonal())

        for i in range(2**self.spin_times_site):
            self.P[i, pnum_diag_sorted[i]] = 1.
        self.P = self.P.tocsc()
        for ii in range(self.annihilators.shape[0]):
            self.annihilators[ii] = self.P.dot(
                self.annihilators[ii].dot(self.P.transpose()))
            self.creators[ii] = self.P.dot(
                self.creators[ii].dot(self.P.transpose()))

        self.N = self.P.dot(self.N.dot(self.P.transpose()))

        # Pascal index subspace index for given particle number
        self.pascal_indices = []
        index = 0
        for ii in range(self.spin_times_site + 1):
            index += int(binom(self.spin_times_site, ii))
            self.pascal_indices.append(index)

        if not spinless:
            # Set up total spin dependend particle number
            self.N_do = sparse.csc_matrix(
                (4**self.nsite, 4**self.nsite))
            self.N_up = sparse.csc_matrix(
                (4**self.nsite, 4**self.nsite))

            for ii in range(self.nsite):
                self.N_do += self.cdag(ii, "do").dot(self.c(ii, "do"))
                self.N_up += self.cdag(ii, "up").dot(self.c(ii, "up"))

            self.N_up_do = np.array(
                [(n_up, n_do) for n_up, n_do in zip(self.N_up.diagonal(),
                                                    self.N_do.diagonal())])

    # define functions to get certain operators
    # notation
    #   do - spin down
    #   up - spin up

    def c(self, ii, spin=None):
        """Returns the annihilation operator at site/orbital 'ii' and with
        spin 'spin'

        Parameters
        ----------
        ii : int
            site/orbital index
        spin : string, optional
            Spin index 'up' or 'do' (down) for spin 1/2 fermions, by default
            None. In case of spinless fermions the argument doesn't need to
            be supplied. If it is supplied, the annihilation operator at
            site/orbital index i is returned.


        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Annihilation operator of site/orbital index 'ii' and spin index
            'spin'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound
        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if (ii > self.nsite - 1):
            raise IndexError('ERROR: index out of bound!')

        if self.spinless:
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.annihilators[ii]
        else:
            if spin == "up":
                return self.annihilators[2 * ii]
            elif spin == "do":
                return self.annihilators[2 * ii + 1]
            else:
                raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def cdag(self, ii, spin=None):
        """Returns the creation operator at site/orbital 'ii' and with
        spin 'spin'

        Parameters
        ----------
        ii : int
            site/orbital index
        spin : string, optional
            Spin index 'up' or 'do' (down) for spin 1/2 fermions, by default
            None. In case of spinless fermions the argument doesn't need to
            be supplied. If it is supplied, the creation operator at
            site/orbital index i is returned.


        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Creation operator of site/orbital index 'ii' and spin index
            'spin'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound
        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if (ii > self.nsite - 1):
            raise IndexError('ERROR: index out of bound!')
        if self.spinless:
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.creators[ii]
        else:
            if spin == "up":
                return self.creators[2 * ii]
            elif spin == "do":
                return self.creators[2 * ii + 1]
            else:
                raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def n(self, ii, spin=None, nelec=None):
        if (ii > self.nsite - 1):
            print('ERROR: index out of bound!')
            exit()

        n = self.cdag(ii, spin).dot(self.c(ii, spin))

        if nelec is None:
            return n
        elif nelec == 0:
            return n[:1, :1]
        else:
            return n[self.pascal_indices[nelec - 1]:self.pascal_indices[nelec],
                     self.pascal_indices[nelec - 1]:self.pascal_indices[nelec]]

###############################################################################
# %%


class BosonicFockOperators:

    def __init__(self, nmodes, nb_max):
        """Class of creation and annihilation for spinless bosons in Fock space

        The bosonic Fock space consists of a number of modes (nmodes) with the
        same cutoff maximum particle number (nb_max). Due to the cutoff of the
        Fock space normal ordering has to be mantained when constructing
        operators out of the creation and annihilation operators.

        Parameters
        ----------
        nmodes : int
            Number of different bosonic modes
        nb_max : int
            Largest number of phonon in a mode
        """
        # number of bosonic modes
        self.nmodes = nmodes

        # largest number of photons in a mode, photonic cut-off

        self.nb_max = nb_max + 1  # +1 accounts for the vacuum

        # setting up creation and annihilation matrices for one mode
        create = sparse.lil_matrix((self.nb_max, self.nb_max))
        destroy = sparse.lil_matrix((self.nb_max, self.nb_max))

        create.setdiag(values=[np.sqrt(i)
                       for i in range(1, self.nb_max + 1)], k=-1)
        create = create.tocsc()
        destroy.setdiag(values=[np.sqrt(i)
                        for i in range(1, self.nb_max + 1)], k=1)
        destroy = destroy.tocsc()
        unit_nb_max = sparse.eye(self.nb_max,
                                 dtype=complex, format="csc")

        # Jordan-Wigner type construction
        # bosonic operators
        self.annihilators = []
        self.creators = []

        self.annihilators = np.full((self.nmodes,), sparse.csc_matrix(
            (self.nb_max**self.nmodes, self.nb_max**self.nmodes)))

        self.creators = np.full((self.nmodes,), sparse.csc_matrix(
            (self.nb_max**self.nmodes, self.nb_max**self.nmodes)))

        for ii in range(self.nmodes):
            b_temp = 1
            bdag_temp = 1

            for jj in range(self.nmodes):
                if (jj == ii):
                    b_temp = sparse.kron(destroy, b_temp, format="csc")
                    bdag_temp = sparse.kron(create, bdag_temp, format="csc")
                else:
                    b_temp = sparse.kron(unit_nb_max, b_temp, format="csc")
                    bdag_temp = sparse.kron(
                        unit_nb_max, bdag_temp, format="csc")

            self.annihilators[ii] = b_temp
            self.creators[ii] = bdag_temp

    def b(self, ii=0):
        """Returns the annihilation operator at mode 'ii'

        Parameters
        ----------
        ii : int, optional
            mode index, by default set to 0

        Returns
        -------
        out: scipy.sparse.csc_matrix (self.nb_max**self.nmodes,
                                      self.nb_max**self.nmodes)
            Annihilation operator at mode 'ii''.

        Raises
        ------
        IndexError
            If mode index is out of bound
        """
        if (ii > self.nmodes - 1):
            print('ERROR: index out of bound!')
            exit()

        return self.annihilators[ii]

    def bdag(self, ii=0):
        """Returns the creation operator at mode 'ii'

        Parameters
        ----------
        ii : int, optional
            mode index, by default set to 0

        Returns
        -------
        out: scipy.sparse.csc_matrix (self.nb_max**self.nmodes,
                                      self.nb_max**self.nmodes)
            Creation operator at mode 'ii''.

        Raises
        ------
        IndexError
            If mode index is out of bound
        """
        if (ii > self.nmodes - 1):
            print('ERROR: index out of bound!')
            exit()

        return self.creators[ii]


# %%
if __name__ == "__main__":
    nsite = 3
    f_op = FermionicFockOperators(nsite)
    identity = sparse.eye(4**nsite, dtype=complex)
    for i in range(nsite):
        for j in range(nsite):
            for s1 in ["up", "do"]:
                for s2 in ["up", "do"]:
                    # purely fock creation and annihilation operators
                    anti_commutation_c_cdag = (f_op.cdag(i, s1)
                                               * f_op.c(j, s2)
                                               + f_op.c(j, s2)
                                               * f_op.cdag(i, s1))
                    anti_commutation_c_c = (f_op.c(i, s1)
                                            * f_op.c(j, s2)
                                            + f_op.c(j, s2)
                                            * f_op.c(i, s1))
                    anti_commutation_cdag_cdag = (f_op.cdag(i, s1)
                                                  * f_op.cdag(j, s2)
                                                  + f_op.cdag(j, s2)
                                                  * f_op.cdag(i, s1))
                    if i == j and s1 == s2:
                        assert (anti_commutation_c_cdag -
                                identity).count_nonzero() == 0

                    else:
                        assert (
                            anti_commutation_c_cdag).count_nonzero() == 0
                        assert (anti_commutation_c_c).count_nonzero() == 0
                        assert (
                            anti_commutation_cdag_cdag).count_nonzero() == 0
# TODO: Include example for bosonic operators
# TODO: Include tests
# %%
