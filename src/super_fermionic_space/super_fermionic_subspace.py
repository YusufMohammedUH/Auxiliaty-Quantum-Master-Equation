from typing import List, Tuple, Union
import numpy as np
from scipy import sparse
import src.super_fermionic_space.define_super_fermionic_operators as sf_op


def add_spin_sectors(sector1: Tuple[int, int], sector2: Tuple[int, int]
                     ) -> Tuple[int, int]:
    """Add to tuples of lenght 2, by adding each index together.

    Parameters
    ----------
    sector1 : Tuple
        First tuple of length 2
    sector2 : Tuple
        Second tuple of length 2

    Returns
    -------
    Tuple
        Added tuple of length 2
    """
    return (sector1[0] + sector2[0], sector1[1] + sector2[1])


def get_subspace_object(object: sparse.csc_matrix,
                        permutation_op_left: Union[
                            sparse.csc_matrix, None] = None,
                        permutation_op_right: Union[
                            sparse.csc_matrix, None] = None
                        ) -> sparse.csc_matrix:
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
    if permutation_op_left is None:
        if object.shape[0] != 1:
            raise ValueError("If no permutation_op_left is passed,"
                             + "Object has to be a vector of shape"
                             + " (1, dim).")
        if permutation_op_right is None:
            raise ValueError("A permutation operator has to be passed.")
        return (object * permutation_op_right[1].transpose()
                )[0, :permutation_op_right[0]]
    elif permutation_op_right is None:
        if object.shape[1] != 1:
            raise ValueError("If no permutation_op_right is passed, Object has"
                             + " to be a vector of shape (dim, 1).")
        return (permutation_op_left[1] * object
                )[:permutation_op_left[0], 0]
 
    return (permutation_op_left[1] * object
            * permutation_op_right[1].transpose()
            )[:permutation_op_left[0],
                :permutation_op_right[0]]


class SubspaceDecomposition(sf_op.SuperFermionicOperators):
    """Super-fermionic child class with additional definition of projectors and
    the ability to define permutation operators. This permutation operators can
    be used to project operators and vectors of the full super-fermionic
    space in a desired subspace.


    Parameters
    ----------
    parent : sf_op.SuperFermionicOperators
        Parent class
    """

    def particle_number_fock_subspace_projector(self, nelec: int
                                                ) -> sparse.csc_matrix:
        """Projector for given particle number nelec in Fock space in the
        super-fermionic representation.

        If the projector is applied to an Liouville space operator,
        all elements constructed from Fock spaces with differend particle
        number are traced out.

        This operator should only be used if the dissipation in the lindbladian
        preserves the particle number.

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
        pnum_projector = sparse.csc_matrix(
            (4**(self.fock_ops.spin_times_site),
             4**(self.fock_ops.spin_times_site)), dtype=np.complex128)

        for n in pnum_index:
            n_vector = sparse.lil_matrix(
                (2**self.fock_ops.spin_times_site, 1), dtype=np.complex128)
            n_vector[n, 0] = 1.0

            for m in pnum_index:
                m_vector = sparse.lil_matrix(
                    (2**self.fock_ops.spin_times_site, 1), dtype=np.complex128)
                m_vector[m, 0] = 1
                nm_vector = sparse.kron(n_vector, m_vector, format="csc")
                pnum_projector += nm_vector * nm_vector.transpose()

        return pnum_projector

    def get_permutation_operator(self, indices: np.ndarray, full: bool = False
                                 ) -> Tuple[int, sparse.csc_matrix]:
        """Returns a permutation operator, permuting desired sectors to the
        upper left corner of a the liouville space matrix.

        This can be used to reduce the Lindbladian to the relevant subspace,
        by supplying the indices associated with the values of a conserved
        observable, e.g. in case of particle number conservation the indices of
        corresponding to the a desired particle number in the main diagonal of
        the particle number operator.

        This can be used to accelerate calculations such as the exact
        diagonalization and time propagation.

        Parameters
        ----------
        indices : numpy.ndarray (dim,)
            Indices corresponding to the position of

        full : bool, optional
            If False it returns the permutation operator that permutes the
            relevant sector to the upper left of the matrix and projects
            out the rest. If True the full permutation operator, which doesn't
            project out the rest is returned. By default False

        Returns
        -------
        out: Tuple[int,scipy.sparse.csc_matrix (dim, dim)]
            Dimension of permutation operator and permutation operator for the
            desired sector.
        """

        indices = np.sort(indices)
        dim_subspace = indices.shape[0]
        total_index = np.linspace(0.,
                                  4**(self.fock_ops.spin_times_site) - 1.,
                                  num=4**(self.fock_ops.spin_times_site),
                                  dtype=int)
        n_prime = total_index[dim_subspace:]
        m_prime = np.setdiff1d(total_index, indices)
        mask_n_m_prime_same = np.in1d(n_prime, m_prime)
        n_m_prime_same = n_prime[mask_n_m_prime_same]
        n_prime_diff = np.setdiff1d(n_prime, m_prime)
        m_prime_diff = np.setdiff1d(m_prime, n_prime)

        perm_op_sector = sparse.lil_matrix(
            (4**(self.fock_ops.spin_times_site),
                4**(self.fock_ops.spin_times_site)
             ), dtype=np.complex128)

        for n in range(dim_subspace):
            perm_op_sector[n, indices[n]] = 1.0

        if full:
            for n, m in zip(n_prime_diff, m_prime_diff):
                perm_op_sector[n, m] = 1.0

            for n in n_m_prime_same:
                perm_op_sector[n, n] = 1.0

        return dim_subspace, perm_op_sector.tocsc()

    def particle_number_fock_subspace_permutation_operator(
            self, nelec: int, full: bool = False
    ) -> Tuple[int, sparse.csc_matrix]:
        """Returns a permutation operator, permuting desired Fock particle
        number sector "nelec" to the upper left corner of a the liouville
        space matrix.

        This subspace corresponds to constructing the Liouville space operator
        from the corresponding Fock space operator in the "nelec" particle
        number Fock subspace.

        This can be used to reduce the Lindbladian to the relevant fock
        particle number sector.

        !!!Warning: Use only in systems with particle number conservation!!!

        Parameters
        ----------
        nelec : int
            Particle number

        full : bool, optional
            If False it returns the permutation operator that permutes the
            relevant spin sector to the upper left of the matrix and projects
            out the rest. If True the full permutation operator, which doesn't
            project out the rest is returned. By default False

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Dimension of permutation operator and permutation operator for the
            desired Fock particle number sector "nelec".
        """
        pnum_fock = self.particle_number_fock_subspace_projector(nelec)
        pnum_fock_index = np.where(
            pnum_fock.diagonal() == 1)[0]
        return self.get_permutation_operator(pnum_fock_index, full)


class SpinSectorDecomposition(SubspaceDecomposition):
    r"""SuperFermionicOperators(nsite: int, spin_sector_max: int,
                 target_sites: Union[List, None] = None,
                 spinless: bool = False,
                 tilde_conjugationrule_phase: bool = True) -> None

    SubspaceDecomposition child class decomposes the super-fermionic
    creation and annihilation in spin sectors (\Delta N_{up},\Delta N_{do}).
    In case of spinless fermions the sectors are given by
    (\Delta N=N-N_{tilde})

    Parameters
    ----------
    parent : sf_op.SubspaceDecomposition
        Parent class
    """

    def __init__(self, nsite: int, spin_sector_max: int,
                 target_sites: Union[List, None] = None,
                 spinless: bool = False,
                 tilde_conjugationrule_phase: bool = True) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        assert spin_sector_max >= 0
        SubspaceDecomposition.__init__(
            self, nsite=nsite, spinless=spinless,
            tilde_conjugationrule_phase=tilde_conjugationrule_phase)
        self.spin_sector_max = spin_sector_max
        if spinless:
            self.operator_sectors = {"cdag": 1, 'c': -1, "c_tilde": 1,
                                     "cdag_tilde": -1}
        else:
            self.operator_sectors = {"cdag": {"up": (1, 0), 'do': (0, 1)},
                                     'c_tilde': {"up": (1, 0), 'do': (0, 1)},
                                     'c': {'up': (-1, 0), 'do': (0, -1)},
                                     'cdag_tilde': {'up': (-1, 0),
                                                    'do': (0, -1)},
                                     'n_channel': {'ch': (0, 0),
                                                   'x': [(-1, 1), (1, -1)],
                                                   'y': [(-1, 1), (1, -1)],
                                                   'z': (0, 0)},
                                     'n_channel_tilde': {'ch': (0, 0),
                                                         'x': [(-1, 1),
                                                               (1, -1)],
                                                         'y': [(-1, 1),
                                                               (1, -1)],
                                                         'z': (0, 0)}}

        # TODO: default self.target_sites should in general be a list of all
        #       sites only special cases need a specific value
        if target_sites is None:
            self.target_sites = [
                int((self.fock_ops.nsite - 1) / 2)]
        else:
            self.target_sites = target_sites

        self.projectors = {}
        self.spin_sector_fermi_ops = {site: {'c': {}, 'c_tilde': {},
                                      'cdag': {}, 'cdag_tilde': {}} for site in
                                      self.target_sites}
        self.spin_sectors = None

        self.set_spin_subspace()

    def get_left_sector(self, right_sector: Union[int, Tuple], operator: str,
                        spin: str = None) -> Union[int, Tuple]:
        """Starting from a right spin sector 'right_sector' the final spin
        sector after acting with a super-fermionic annihilation or creation
        operator is returned.

        Parameters
        ----------
        left_sector : Union[int, Tuple]
            Spin sector (N_up - N_up_tilde, N_do - N_do_tilde) if spinful
            fermions or difference N - N_tilde if spinless fermions

        operator : str
            Annihilation/creation operator in super-fermionic space 'cdag',
            'c','cdag_tilde' or 'c_tilde'

        spin : str, optional
            Spin can be 'up','do' or None, by default None

        Returns
        -------
        Union[int, Tuple]
            Sector to the right
        """
        if self.fock_ops.spinless:
            return self.operator_sectors[operator] + right_sector

        return add_spin_sectors(self.operator_sectors[operator][spin],
                                    right_sector)

    def spin_sector_projector(self, sector: Union[Tuple[int, int], int]
                              ) -> sparse.csc_matrix:
        """Projector for given spin sector "sector" in the liouville space.


        Parameters
        ----------
        sector : int or tuple (up, do)
            Spin sector defined by the difference between particles in
            "normal" and "tilde" space for spin up and down if spinful and
            the difference between total particle number in "normal" and
            "tilde" space if spinless.

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Projector for the desired Spin sector subspace.
        """
        if not self.fock_ops.spinless:
            pnum_up_index = np.where(
                self.Delta_N_up.diagonal(
                ) == sector[0])[0]
            pnum_do_index = np.where(
                self.Delta_N_do.diagonal(
                ) == sector[1])[0]
            mask_section = np.in1d(pnum_up_index, pnum_do_index)
            pnum_index = pnum_up_index[mask_section]
            pnum_per_spin_projector = sparse.lil_matrix(
                (4**(self.fock_ops.spin_times_site),
                    4**((self.fock_ops
                         ).spin_times_site)), dtype=np.complex128)

            for n in pnum_index:
                pnum_per_spin_projector[n, n] = 1.0
            pnum_per_spin_projector = pnum_per_spin_projector.tocsc()

            return pnum_per_spin_projector

        pnum_index = np.where(
            (self.N - self.N_tilde).diagonal(
            ) == sector)[0]

        pnum_per_spin_projector = sparse.lil_matrix(
            (4**(self.fock_ops.spin_times_site),
                4**((self.fock_ops
                     ).spin_times_site)), dtype=np.complex128)

        for n in pnum_index:
            pnum_per_spin_projector[n, n] = 1.0
        pnum_per_spin_projector = pnum_per_spin_projector.tocsc()

        return pnum_per_spin_projector

    def spin_sector_permutation_operator(self, sector: Union[
            Tuple[int, int], int], full: bool = False
    ) -> Tuple[int, sparse.csc_matrix]:
        """Returns a permutation operator, permuting desired spin sector
        "sector" to the upper left corner of a the liouville space matrix.

        This can be used to reduce the Lindbladian to the relevant spin
        sectors. And accelerating calculations such as the exact
        diagonalization and time propagation.

        Parameters
        ----------
        sector : int or tuple (up, do)
            Spin sector defined by the difference between particles in
            "normal" and "tilde" space for spin up and down is spinful and
            total particle number difference between "normal" and "tilde" space
            if spinless.

        full : bool, optional
            If False it returns the permutation operator that permutes the
            relevant spin sector to the upper left of the matrix and projects
            out the rest. If True the full permutation operator, which doesn't
            project out the rest is returned. By default False

        Returns
        -------
        out: scipy.sparse.csc_matrix (dim, dim)
            Permutation operator for the desired Spin sector subspace.
        """
        if not self.fock_ops.spinless:
            pnum_up_index = np.where(
                self.Delta_N_up.diagonal(
                ) == sector[0])[0]
            pnum_do_index = np.where(
                self.Delta_N_do.diagonal(
                ) == sector[1])[0]
            mask_section = np.in1d(pnum_up_index, pnum_do_index)
            pnum_index = np.sort(pnum_up_index[mask_section])
        else:
            pnum_index = np.where((self.N - self.N_tilde).diagonal(
            ) == sector)[0]
            pnum_index = np.sort(pnum_index)

        return self.get_permutation_operator(
            pnum_index, full)

    def set_possible_spin_sectors(self) -> None:
        """Calculate all relevant spin sectors, that can be reached with the
        highest, desired correlator. Set all possible sectors as attribute of
        the object.

        E.G. a four point vertex can access all spin sector , (i,j)
        with i<=2, j<=2 and |i-j|<=2
        """
        sector_range = np.arange(-self.spin_sector_max,
                                 self.spin_sector_max + 1)
        if not self.fock_ops.spinless:
            spin_sectors = np.array(
                np.meshgrid(sector_range, sector_range)).T.reshape(-1, 2)
            allowed_combinations = np.sum(np.abs(spin_sectors), -1)
            allowed_combinations = allowed_combinations <= self.spin_sector_max
            self.spin_sectors = spin_sectors[allowed_combinations]
            self.spin_sectors = [tuple(x) for x in self.spin_sectors]
        else:
            self.spin_sectors = sector_range

    def set_spin_sectors_permutation_ops(self) -> None:
        """Calculate and store all perutation operators projecting on to
        the relevant spin sectors in self.projectors.
        """

        self.projectors = {}
        for com in self.spin_sectors:
            if not self.fock_ops.spinless:
                self.projectors[tuple(com)] = \
                    self.spin_sector_permutation_operator(com)
            else:
                self.projectors[com] = \
                    self.spin_sector_permutation_operator(com)

    def set_spin_sectors_fermionic_ops(self) -> None:
        """Calculate and store the fermionic operators within the accessible
        spin sectors subspace. The spin sectors are stored in
        self.spin_sectors. This reduces the dimension of the operators in
        super-fermionic space.
        """

        self.spin_sector_fermi_ops = {site: {'c': {}, 'c_tilde': {},
                                      'cdag': {}, 'cdag_tilde': {},
                                             'n_channel': {},
                                             'n_channel_tilde': {}}
                                      for site in self.target_sites}
        if not self.fock_ops.spinless:
            n_channel = {'ch': {}, 'x': {}, 'y': {}, 'z': {}}
            n_channel_tilde = {'ch': {}, 'x': {}, 'y': {}, 'z': {}}
            cdag_up_sector = {}
            cdag_up_tilde_sector = {}
            c_up_sector = {}
            c_up_tilde_sector = {}
            cdag_do_sector = {}
            cdag_do_tilde_sector = {}
            c_do_sector = {}
            c_do_tilde_sector = {}
        else:
            cdag_sector = {}
            cdag_tilde_sector = {}
            c_sector = {}
            c_tilde_sector = {}

        for site in self.target_sites:
            if not self.fock_ops.spinless:
                for sector in self.spin_sectors:
                    sector = tuple(sector)
                    up_plus = add_spin_sectors((1, 0), sector)
                    up_minus = add_spin_sectors((-1, 0), sector)
                    do_plus = add_spin_sectors((0, 1), sector)
                    do_minus = add_spin_sectors((0, -1), sector)
                    n_channel_up_up = (add_spin_sectors((1, 0), up_minus),
                                       sector)
                    n_channel_up_do = (add_spin_sectors((1, 0), do_minus),
                                       sector)
                    n_channel_do_up = (add_spin_sectors((0, 1), up_minus),
                                       sector)
                    n_channel_do_do = (add_spin_sectors((0, 1), do_minus),
                                       sector)

                    if n_channel_up_up[0] in self.spin_sectors:
                        n_channel['ch'][n_channel_up_up] = get_subspace_object(
                            self.n_channel(site, 'ch'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])
                        n_channel['x'][n_channel_up_up] = get_subspace_object(
                            self.n_channel(site, 'x'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])
                        n_channel['y'][n_channel_up_up] = get_subspace_object(
                            self.n_channel(site, 'y'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])
                        n_channel['z'][n_channel_up_up] = get_subspace_object(
                            self.n_channel(site, 'z'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])

                        n_channel_tilde['ch'][n_channel_up_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'ch'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])
                        n_channel_tilde['x'][n_channel_up_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'x'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])
                        n_channel_tilde['y'][n_channel_up_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'y'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])
                        n_channel_tilde['z'][n_channel_up_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'z'), self.projectors[
                                n_channel_up_up[0]], self.projectors[
                                    n_channel_up_up[1]])

                    if n_channel_up_do[0] in self.spin_sectors:
                        n_channel['ch'][n_channel_up_do] = get_subspace_object(
                            self.n_channel(site, 'ch'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])
                        n_channel['x'][n_channel_up_do] = get_subspace_object(
                            self.n_channel(site, 'x'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])
                        n_channel['y'][n_channel_up_do] = get_subspace_object(
                            self.n_channel(site, 'y'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])
                        n_channel['z'][n_channel_up_do] = get_subspace_object(
                            self.n_channel(site, 'z'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])

                        n_channel_tilde['ch'][n_channel_up_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'ch'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])
                        n_channel_tilde['x'][n_channel_up_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'x'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])
                        n_channel_tilde['y'][n_channel_up_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'y'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])
                        n_channel_tilde['z'][n_channel_up_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'z'), self.projectors[
                                n_channel_up_do[0]], self.projectors[
                                    n_channel_up_do[1]])

                    if n_channel_do_up[0] in self.spin_sectors:
                        n_channel['ch'][n_channel_do_up] = get_subspace_object(
                            self.n_channel(site, 'ch'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])
                        n_channel['x'][n_channel_do_up] = get_subspace_object(
                            self.n_channel(site, 'x'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])
                        n_channel['y'][n_channel_do_up] = get_subspace_object(
                            self.n_channel(site, 'y'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])
                        n_channel['z'][n_channel_do_up] = get_subspace_object(
                            self.n_channel(site, 'z'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])

                        n_channel_tilde['ch'][n_channel_do_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'ch'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])
                        n_channel_tilde['x'][n_channel_do_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'x'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])
                        n_channel_tilde['y'][n_channel_do_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'y'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])
                        n_channel_tilde['z'][n_channel_do_up] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'z'), self.projectors[
                                n_channel_do_up[0]], self.projectors[
                                    n_channel_do_up[1]])

                    if n_channel_do_do[0] in self.spin_sectors:
                        n_channel['ch'][n_channel_do_do] = get_subspace_object(
                            self.n_channel(site, 'ch'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])
                        n_channel['x'][n_channel_do_do] = get_subspace_object(
                            self.n_channel(site, 'x'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])
                        n_channel['y'][n_channel_do_do] = get_subspace_object(
                            self.n_channel(site, 'y'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])
                        n_channel['z'][n_channel_do_do] = get_subspace_object(
                            self.n_channel(site, 'z'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])

                        n_channel_tilde['ch'][n_channel_do_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'ch'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])
                        n_channel_tilde['x'][n_channel_do_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'x'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])
                        n_channel_tilde['y'][n_channel_do_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'y'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])
                        n_channel_tilde['z'][n_channel_do_do] = \
                            get_subspace_object(self.n_channel_tilde(
                                site, 'z'), self.projectors[
                                n_channel_do_do[0]], self.projectors[
                                    n_channel_do_do[1]])

                    if up_plus in self.spin_sectors:
                        cdag_up_sector[(up_plus, sector)] = \
                            get_subspace_object(
                            self.cdag(site, 'up'), self.projectors[up_plus],
                            self.projectors[sector])

                        c_up_tilde_sector[(up_plus, sector)] = \
                            get_subspace_object(self.c_tilde(site, 'up'),
                                                self.projectors[up_plus],
                                                self.projectors[sector])

                    if up_minus in self.spin_sectors:
                        c_up_sector[(up_minus, sector)] = get_subspace_object(
                            self.c(site, 'up'),
                            self.projectors[up_minus], self.projectors[sector])

                        cdag_up_tilde_sector[(up_minus, sector)] = \
                            get_subspace_object(self.cdag_tilde(site, 'up'),
                                                self.projectors[up_minus],
                                                self.projectors[sector])

                    if do_plus in self.spin_sectors:
                        cdag_do_sector[(do_plus, sector)] = \
                            get_subspace_object(self.cdag(site, 'do'),
                                                self.projectors[do_plus],
                                                self.projectors[sector])
                        c_do_tilde_sector[(do_plus, sector)] = \
                            get_subspace_object(self.c_tilde(site, 'do'),
                                                self.projectors[do_plus],
                                                self.projectors[sector])

                    if do_minus in self.spin_sectors:
                        c_do_sector[(do_minus, sector)] = \
                            get_subspace_object(self.c(site, 'do'),
                                                self.projectors[do_minus],
                                                self.projectors[sector])
                        cdag_do_tilde_sector[(do_minus, sector)] = \
                            get_subspace_object(self.cdag_tilde(site, 'do'),
                                                self.projectors[do_minus],
                                                self.projectors[sector])

                self.spin_sector_fermi_ops[site]['c']['up'] = c_up_sector
                self.spin_sector_fermi_ops[site]['c']['do'] = c_do_sector
                self.spin_sector_fermi_ops[site]['c_tilde']['up'] = \
                    c_up_tilde_sector
                self.spin_sector_fermi_ops[site]['c_tilde']['do'] = \
                    c_do_tilde_sector

                self.spin_sector_fermi_ops[site]['cdag']['up'] = cdag_up_sector
                self.spin_sector_fermi_ops[site]['cdag']['do'] = cdag_do_sector
                self.spin_sector_fermi_ops[site]['cdag_tilde']['up'] = \
                    cdag_up_tilde_sector
                self.spin_sector_fermi_ops[site]['cdag_tilde']['do'] = \
                    cdag_do_tilde_sector

                self.spin_sector_fermi_ops[site]['n_channel'] = n_channel
                self.spin_sector_fermi_ops[site]['n_channel_tilde'] = \
                    n_channel_tilde
            else:
                for sector in self.spin_sectors:
                    plus = sector + 1
                    minus = sector - 1

                    if plus in self.spin_sectors:
                        cdag_sector[(plus, sector)] = get_subspace_object(
                            self.cdag(site), self.projectors[plus],
                            self.projectors[sector])
                        c_tilde_sector[(plus, sector)] = get_subspace_object(
                            self.c_tilde(site), self.projectors[plus],
                            self.projectors[plus])

                    if minus in self.spin_sectors:
                        c_sector[(minus, sector)] = get_subspace_object(
                            self.c(site), self.projectors[minus],
                            self.projectors[sector])
                        cdag_tilde_sector[(minus, sector)] = \
                            get_subspace_object(
                            self.cdag_tilde(site),
                            self.projectors[minus],
                            self.projectors[sector])

                self.spin_sector_fermi_ops[site]['c'] = c_sector
                self.spin_sector_fermi_ops[site]['c_tilde'] = c_tilde_sector

                self.spin_sector_fermi_ops[site]['cdag'] = cdag_sector
                self.spin_sector_fermi_ops[site]['cdag_tilde'] = \
                    cdag_tilde_sector

    def set_spin_subspace(self) -> None:
        """Set up all possible spin sectors, calculate the permutation operators
        in order to project out the spin sectors and calculate the projected
        super-fermionic operators.
        """
        self.set_possible_spin_sectors()
        self.set_spin_sectors_permutation_ops()
        self.set_spin_sectors_fermionic_ops()

    def get_subspace_object(self, object: sparse.csc_matrix,
                            sector_left: Union[Tuple[int, int], int] = None,
                            sector_right: Union[Tuple[int, int], int] = None
                            ) -> sparse.csc_matrix:
        """Extract an Liouville space operator or vector in the desired spin
        sector subspace.


        Parameters
        ----------
        object : sparse.csc_matrix (dim, 1), (1, dim) or (dim, dim)
            Super-fermionic space operator or vector.

        sector_left : Union[Tuple[int, int], int], optional
            Left spin sector, by default None

        sector_right : Union[Tuple[int, int], int], optional
            Right spin sector, by default None

        Returns
        -------
        out: sparse.csc_matrix( dim, dim)
        Object in subspace connecting, which connects the spin sectors
        sector_left and sector_right.
        """

        assert not ((sector_right is None) and (sector_left is None))
        if sector_left is None:
            return get_subspace_object(
                object=object,
                permutation_op_right=self.projectors[sector_right])
        elif sector_right is None:
            return get_subspace_object(
                object=object, permutation_op_left=self.projectors[sector_left]
            )

        return get_subspace_object(
            object=object,
            permutation_op_left=self.projectors[sector_left],
            permutation_op_right=self.projectors[sector_right])

    def cdag_sector(self, sector: Union[Tuple[int, int],
                                        Tuple[Tuple[int, int]]], site: int,
                    spin: Union[str, None] = None) -> sparse.csc_matrix:
        r"""Returns the 'normal' space creation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : Union[Tuple[int, int], Tuple[Tuple[int, int]]]
            Spin sector of interest. If spinless fermions, than use a tuple of
            two integers.

            The sector is than given by a fixed value for
            (\Delta N_left,\Delta N_right).
            with
            \Delta N = N-N_tilde

            If spinful fermions, the sector is than given by a tuple of tuple
            of two integers:
             ((\Delta N_{up},\Delta N_{do})_left,
             (\Delta N_{up},\Delta N_{do})_right), e.g. ((0,1),(0,0)).

        site : int
            site/orbital index

        spin : Union[str, None], optional
            Spin index 'up' or 'do' (down) for spin 1/2 fermions, by default
            None. In case of spinless fermions the argument doesn't need to
            be supplied. If it is supplied, the creation operator at
            site/orbital 'site' is returned.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Creation operator of site/orbital index 'site' and spin index
            'spin' in sector 'sector'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound

        IndexError
            If sector index is out of bound in spinless system

        IndexError
            If sector index is out of bound in spinful system

        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if site not in self.target_sites:
            raise IndexError('ERROR: Index out of bound!')

        if self.fock_ops.spinless:
            if np.abs(sector) > self.spin_sector_max:
                raise IndexError("ERROR: Sector out of bound!")
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")

            return self.spin_sector_fermi_ops[site]['cdag'][sector]

        abs_sector = np.array([sum(np.abs(s)) for s in sector])
        if np.any(abs_sector > self.spin_sector_max):
            raise IndexError("ERROR: Sector out of bound!")
        if spin == "up":
            if (sector[0][1] != sector[1][1]) or (
                    sector[0][0] - 1 != sector[1][0]):
                raise IndexError("ERROR: Sector can't be reached!")
            return self.spin_sector_fermi_ops[site]['cdag']['up'][sector]
        elif spin == "do":
            if (sector[0][0] != sector[1][0]) or (
                    sector[0][1] - 1 != sector[1][1]):
                raise IndexError("ERROR: Sector can't be reached!")

            return self.spin_sector_fermi_ops[site]['cdag']['do'][sector]

        raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def c_sector(self, sector: Union[Tuple[int, int], Tuple[Tuple[int, int]]],
                 site: int, spin: Union[str, None] = None
                 ) -> sparse.csc_matrix:
        r"""Returns the 'normal' space annihilation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : Union[Tuple[int, int], Tuple[Tuple[int, int]]]
            Spin sector of interest. If spinless fermions, than use a tuple of
            two integers.

            The sector is than given by a fixed value for
            (\Delta N_left,\Delta N_right).
            with
            \Delta N = N-N_tilde

            If spinful fermions, the sector is than given by a tuple of tuple
            of two integers:
             ((\Delta N_{up},\Delta N_{do})_left,
             (\Delta N_{up},\Delta N_{do})_right), e.g. ((0,-1),(0,0)).

        site : int
            site/orbital index

        spin : Union[str, None], optional
            Spin index 'up' or 'do' (down) for spin 1/2 fermions, by default
            None. In case of spinless fermions the argument doesn't need to
            be supplied. If it is supplied, the annihilation operator at
            site/orbital 'site' is returned.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Annihilation operator of site/orbital index 'site' and spin index
            'spin' in sector 'sector'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound

        IndexError
            If sector index is out of bound in spinless system

        IndexError
            If sector index is out of bound in spinful system

        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if site not in self.target_sites:
            raise IndexError('ERROR: index out of bound!')

        if self.fock_ops.spinless:
            if np.abs(sector) > self.spin_sector_max:
                raise IndexError("ERROR: Sector out of bound!")
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.spin_sector_fermi_ops[site]['c'][sector]

        abs_sector = np.array([sum(np.abs(s)) for s in sector])
        if np.any(abs_sector > self.spin_sector_max):
            raise IndexError("ERROR: Sector out of bound!")

        if spin == "up":
            if (sector[0][1] != sector[1][1]) or (
                    sector[0][0] + 1 != sector[1][0]):
                raise IndexError("ERROR: Sector can't be reached!")
            return self.spin_sector_fermi_ops[site]['c']['up'][sector]
        elif spin == "do":
            if (sector[0][0] != sector[1][0]) or (
                    sector[0][1] + 1 != sector[1][1]):
                raise IndexError("ERROR: Sector can't be reached!")

            return self.spin_sector_fermi_ops[site]['c']['do'][sector]

        raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def cdag_tilde_sector(self, sector: Union[Tuple[int, int],
                                              Tuple[Tuple[int, int]]],
                          site: int, spin: Union[str, None] = None
                          ) -> sparse.csc_matrix:
        r"""Returns the 'tilde' space creation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : Union[Tuple[int, int], Tuple[Tuple[int, int]]]
            Spin sector of interest. If spinless fermions, than use a tuple of
            two integers.

            The sector is than given by a fixed value for
            (\Delta N_left,\Delta N_right).
            with
            \Delta N = N-N_tilde

            If spinful fermions, the sector is than given by a tuple of tuple
            of two integers:
             ((\Delta N_{up},\Delta N_{do})_left,
             (\Delta N_{up},\Delta N_{do})_right), e.g. ((0,-1),(0,0)).

        site : int
            site/orbital index

        spin : Union[str, None], optional
            Spin index 'up' or 'do' (down) for spin 1/2 fermions, by default
            None. In case of spinless fermions the argument doesn't need to
            be supplied. If it is supplied, the creation operator at
            site/orbital 'site' is returned.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Creation operator of site/orbital index 'site' and spin index
            'spin' in sector 'sector'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound

        IndexError
            If sector index is out of bound in spinless system

        IndexError
            If sector index is out of bound in spinful system

        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if site not in self.target_sites:
            raise IndexError('ERROR: index out of bound!')

        if self.fock_ops.spinless:
            if np.abs(sector) > self.spin_sector_max:
                raise IndexError("ERROR: Sector out of bound!")
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.spin_sector_fermi_ops[site]['cdag_tilde'][sector]

        abs_sector = np.array([sum(np.abs(s)) for s in sector])
        if np.any(abs_sector > self.spin_sector_max):
            raise IndexError("ERROR: Sector out of bound!")

        if spin == "up":
            if (sector[0][1] != sector[1][1]) or (
                    sector[0][0] + 1 != sector[1][0]):
                raise IndexError("ERROR: Sector can't be reached!")

            return self.spin_sector_fermi_ops[site]['cdag_tilde']['up'][
                sector]
        elif spin == "do":
            if (sector[0][0] != sector[1][0]) or (
                    sector[0][1] + 1 != sector[1][1]):
                raise IndexError("ERROR: Sector can't be reached!")

            return self.spin_sector_fermi_ops[site]['cdag_tilde']['do'][
                sector]
        else:
            raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def c_tilde_sector(self, sector: Union[Tuple[int, int],
                                           Tuple[Tuple[int, int]]], site: int,
                       spin: Union[str, None] = None) -> sparse.csc_matrix:
        r"""Returns the 'tilde' space annihilation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : Union[Tuple[int, int], Tuple[Tuple[int, int]]]
            Spin sector of interest. If spinless fermions, than use a tuple of
            two integers.

            The sector is than given by a fixed value for
            (\Delta N_left,\Delta N_right).
            with
            \Delta N = N-N_tilde

            If spinful fermions, the sector is than given by a tuple of tuple
            of two integers:
             ((\Delta N_{up},\Delta N_{do})_left,
             (\Delta N_{up},\Delta N_{do})_right), e.g. ((0,1),(0,0)).

        site : int
            site/orbital index

        spin : Union[str, None], optional
            Spin index 'up' or 'do' (down) for spin 1/2 fermions, by default
            None. In case of spinless fermions the argument doesn't need to
            be supplied. If it is supplied, the annihilation operator at
            site/orbital 'site' is returned.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Annihilation operator of site/orbital index 'site' and spin index
            'spin' in sector 'sector'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound

        IndexError
            If sector index is out of bound in spinless system

        IndexError
            If sector index is out of bound in spinful system

        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if site not in self.target_sites:
            raise IndexError('ERROR: index out of bound!')

        if self.fock_ops.spinless:
            if np.abs(sector) > self.spin_sector_max:
                raise IndexError("ERROR: Sector out of bound!")
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.spin_sector_fermi_ops[site]['c_tilde'][sector]

        abs_sector = np.array([sum(np.abs(s)) for s in sector])
        if np.any(abs_sector > self.spin_sector_max):
            raise IndexError("ERROR: Sector out of bound!")

        if spin == "up":
            if (sector[0][1] != sector[1][1]) or (
                    sector[0][0] - 1 != sector[1][0]):
                raise IndexError("ERROR: Sector can't be reached!")

            return self.spin_sector_fermi_ops[site]['c_tilde']['up'][
                sector]
        elif spin == "do":
            if (sector[0][0] != sector[1][0]) or (
                    sector[0][1] - 1 != sector[1][1]):
                raise IndexError("ERROR: Sector can't be reached!")

            return self.spin_sector_fermi_ops[site]['c_tilde']['do'][
                sector]

        raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def n_channel_sector(self, sector: Tuple[Tuple[int, int]], site: int,
                         channel: str = 'ch') -> sparse.csc_matrix:
        r"""Returns the 'normal' space charge or spin density operator in
        sector 'sector' at site/orbital 'site'.

        The charge or spin density operator are given by:
            $\rho^{\xi}_i = \sum_{s s'} c^{\dagger}_{i\,s}
            \sigma^{\xi}_{s s'} c_{i\,s'}$

        with
            $\displaystyle \sigma^{\xi}_{s s'} \in  \{\mathds{1}, \sigma^{x},
            \sigma^{y},\sigma^{z} \}$

        Parameters
        ----------
        sector : Tuple[Tuple[int, int]]
            Spin sector of interest. The sector is than given by a
            tuple of tuple of two integers:
             ((\Delta N_{up},\Delta N_{do})_left,
             (\Delta N_{up},\Delta N_{do})_right), e.g. ((0,1),(0,0)).

        site : int
            site/orbital index

        channel : str, optional
            Channel index 'ch','x', 'y' or 'z', by default 'ch'.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Charge or spin density operator at site/orbital of site/orbital
            index 'site' and in channel 'channel' in sector 'sector'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound

        IndexError
            If sector out of bound
        """
        assert len(sector) == 2
        assert not self.fock_ops.spinless
        if site not in self.target_sites:
            raise IndexError('ERROR: index out of bound!')

        abs_sector = np.array([sum(np.abs(s)) for s in sector])
        if np.any(abs_sector > self.spin_sector_max):
            raise IndexError("ERROR: Sector out of bound!")

        return self.spin_sector_fermi_ops[site]['n_channel'][channel][sector]

    def n_channel_tilde_sector(self, sector: Tuple[Tuple[int, int]], site: int,
                               channel: str = 'ch') -> sparse.csc_matrix:
        r"""Returns the 'normal' space charge or spin density operator in
        sector 'sector' at site/orbital 'site'.

        The charge or spin density operator are given by:
            $\rho^{\xi}_i = \sum_{s s'} c^{\dagger}_{i\,s}
            \sigma^{\xi}_{s s'} c_{i\,s'}$

        with
            $\displaystyle \sigma^{\xi}_{s s'} \in  \{\mathds{1}, \sigma^{x},
            \sigma^{y},\sigma^{z} \}$

        Parameters
        ----------
        sector : Tuple[Tuple[int, int]]
            Spin sector of interest. The sector is than given by a
            tuple of tuple of two integers:
             ((\Delta N_{up},\Delta N_{do})_left,
             (\Delta N_{up},\Delta N_{do})_right), e.g. ((0,1),(0,0)).

        site : int
            site/orbital index

        channel : str, optional
            Channel index 'ch','x', 'y' or 'z', by default 'ch'.

        Returns
        -------
        out: scipy.sparse.csc_matrix (2**self.spin_times_site,
                                    2**self.spin_times_site)
            Charge or spin density operator at site/orbital of site/orbital
            index 'site' and in channel 'channel' in sector 'sector'.

        Raises
        ------
        IndexError
            If site/orbital index is out of bound

        IndexError
            If sector out of bound
        """
        assert len(sector) == 2
        assert not self.fock_ops.spinless
        if site not in self.target_sites:
            raise IndexError('ERROR: index out of bound!')

        abs_sector = np.array([sum(np.abs(s)) for s in sector])
        if np.any(abs_sector > self.spin_sector_max):
            raise IndexError("ERROR: Sector out of bound!")

        return self.spin_sector_fermi_ops[site]['n_channel_tilde'][channel][
            sector]


SuperFermionicOperatorType = Union[sf_op.SuperFermionicOperators,
                                   SpinSectorDecomposition,
                                   SubspaceDecomposition]
