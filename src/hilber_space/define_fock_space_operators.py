from typing import Union
import numpy as np
import numpy.typing as npt
from scipy.special import binom
from scipy import sparse


def commutator(a: npt.ArrayLike, b: npt.ArrayLike
               ) -> npt.ArrayLike:
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


def anti_commutator(a: npt.ArrayLike, b: npt.ArrayLike
                    ) -> npt.ArrayLike:
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
    """FermionicFockOperators(nsite: int, spinless: bool = False,
                 sorted_particle_number: bool = True)
    Class of fermiononic creation and annihilation operators in Fock
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

    sorted_particle_number: bool, optional
        The Fock space is sorted in incrising order if set to True, by
        default False.

    Attributes
    ----------
    spinless: bool
        Indicates if fermions are spinless or spin 1/2.

    nsite: int
        Number of sites/ orbitals in the Fermionic problem

    sigma_plus: scipy.sparse.csc_matrix (2,2)
        Pauli sigma plus matrix

    sorted_particle_number: bool
        The Fock space is sorted in incrising order if set to True, by
        default False.

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

    pascal_indices:  list
        List of supspace dimensions with certain particle numbers.

    N_up: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                2**self.spin_times_site)
        Total particle operator of spin up fermions.

    N_do: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                2**self.spin_times_site)
        Total particle operator of spin down fermions.
    channel: List[str]
        Contains the abbreviation for the charge and spin channels used in
        charge and spin channels density operator.
    """

    def __init__(self, nsite: int, spinless: bool = False,
                 sorted_particle_number: bool = True) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        self.channels = ['ch', 'x', 'y', 'z']
        if spinless:
            print("Constructing spinless fermionic Fock space.")
        self.spinless = spinless
        self.nsite = nsite
        self.sorted_particle_number = sorted_particle_number
        # Jordan-Wigner transformation
        # fermionic operators

        # Set up sigma matrices
        sigma_plus = np.zeros((2, 2), dtype=np.complex128)
        sigma_plus[0, 1] = 1.0
        self.sigma_plus = sparse.csc_matrix(sigma_plus)

        sigma_minus = np.zeros((2, 2), dtype=np.complex128)
        sigma_minus[1, 0] = 1.0
        self.sigma_minus = sparse.csc_matrix(sigma_minus)

        sigma_z = np.identity(2, dtype=np.complex128)
        sigma_z[1, 1] = -1.0
        self.sigma_z = sparse.csc_matrix(sigma_z)

        self.unit2 = sparse.eye(2).tocsc()

        sigma_x = np.zeros((2, 2), dtype=np.complex128)
        sigma_x[0, 1] = 1.0
        sigma_x[1, 0] = 1.0
        self.sigma_x = sparse.csc_matrix(sigma_x)

        sigma_y = np.zeros((2, 2), dtype=np.complex128)
        sigma_y[0, 1] = -1.0j
        sigma_y[1, 0] = 1.0j
        self.sigma_y = sparse.csc_matrix(sigma_y)

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
        if self.sorted_particle_number:
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

    # define functions to get certain operators
    # notation
    #   do - spin down
    #   up - spin up

    def c(self, ii: int, spin: Union[str, None] = None) -> sparse.csc_matrix:
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

        if spin == "up":
            return self.annihilators[2 * ii]
        elif spin == "do":
            return self.annihilators[2 * ii + 1]
        else:
            raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def cdag(self, ii: int, spin: Union[str, None] = None
             ) -> sparse.csc_matrix:
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

        if spin == "up":
            return self.creators[2 * ii]
        elif spin == "do":
            return self.creators[2 * ii + 1]
        else:
            raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def n(self, ii: int, spin: Union[str, None] = None, nelec: int = None
          ) -> sparse.csc_matrix:
        """Returns the particle number operator at site/orbital 'ii' and with
        spin 'spin'

        Parameters
        ----------
        ii : int
            site/orbital index

        spin : string, optional
            Spin index 'up' or 'do' (down) for spin 1/2 fermions, by default
            None. In case of spinless fermions the argument doesn't need to
            be supplied. If it is supplied, the particle number operator at
            site/orbital index i is returned.

        nelec : int, optional
            Particle number, by default None. If supplied, the particle
            number operator in the sector with nelec total electrons is
            returned is returned.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Particle number operator of site/orbital index 'ii' and spin index
            'spin'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound

        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if self.sorted_particle_number:
            raise ValueError('ERROR: The Fock space is not sorted!')
        if (ii > self.nsite - 1):
            raise ValueError('ERROR: index out of bound!')

        n = self.cdag(ii, spin).dot(self.c(ii, spin))

        if nelec is None:
            return n
        elif nelec == 0:
            return n[:1, :1]
        else:
            return n[self.pascal_indices[nelec - 1]:self.pascal_indices[nelec],
                     self.pascal_indices[nelec - 1]:self.pascal_indices[nelec]]

    def n_channel(self, ii: int, channel: str = 'ch'
                  ) -> sparse.csc_matrix:
        """Returns the charge or spin density operator at site/orbital
        'ii'. Channel is one of 'ch','x', 'y' or 'z', where 'ch' is the charge
        channel and 'x','y','z' the spin channels. In case of spinless fermions
        the density operator is returned.

        Parameters
        ----------
        ii : int
            site/orbital index

        channel : string, optional
            Channel index 'ch','x', 'y' or 'z', by default 'ch'.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Charge or spin density operator at site/orbital 'ii'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound
        ValueError
            If channel passed is not one of 'ch', 'x', 'y', 'z'.
        """
        if channel not in self.channels:
            raise ValueError('ERROR: Channel passed have to be in' +
                             ' self.channels!')
        if self.spinless:
            print("Calculating charge/spin density operator for spinless " +
                  "fermions")
            return self.n(ii)

        nchannel = sparse.csc_matrix(
            (2**self.spin_times_site, 2**self.spin_times_site))
        tmp = None
        if channel == 'ch' or channel is None:
            tmp = self.unit2
        elif channel == 'x':
            tmp = self.sigma_x
        elif channel == 'y':
            tmp = self.sigma_y
        elif channel == 'z':
            tmp = self.sigma_y

        for i, spin1 in enumerate(['up', 'do']):
            for j, spin2 in enumerate(['up', 'do']):
                if tmp[i, j] != 0:
                    nchannel += tmp[i, j] * self.cdag(
                        ii=ii, spin=spin1).dot(
                        self.c(ii=ii, spin=spin2))
        return nchannel

###############################################################################


class BosonicFockOperators:
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

    def __init__(self, nmodes: int, nb_max: int) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
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
                                 dtype=np.complex128, format="csc")

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

    def b(self, ii: int = 0) -> sparse.csc_matrix:
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
            raise ValueError('ERROR: index out of bound!')

        return self.annihilators[ii]

    def bdag(self, ii: int = 0) -> sparse.csc_matrix:
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
            raise ValueError('ERROR: index out of bound!')

        return self.creators[ii]
