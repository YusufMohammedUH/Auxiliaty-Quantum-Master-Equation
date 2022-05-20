import numpy as np
from scipy import sparse
import src.super_fermionic_space.define_super_fermionic_operators as sf_op


# TODO: SubspaceDecomposition should only contain the function containing the
#      Function get_permutation_operator
# TODO: A subsequent class for Lindbladians which are particle number
#      conserving with the corresponding creation and annihilation operators
#      could be created


def add_spin_sectors(sector1, sector2):
    return (sector1[0] + sector2[0], sector1[1] + sector2[1])


def get_subspace_object(object, permutation_op_left=None,
                        permutation_op_right=None):
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
    else:
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

    def particle_number_fock_subspace_projector(self, nelec):
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
             4**(self.fock_ops.spin_times_site)), dtype=complex)

        for n in pnum_index:
            n_vector = sparse.lil_matrix(
                (2**self.fock_ops.spin_times_site, 1), dtype=complex)
            n_vector[n, 0] = 1.0

            for m in pnum_index:
                m_vector = sparse.lil_matrix(
                    (2**self.fock_ops.spin_times_site, 1), dtype=complex)
                m_vector[m, 0] = 1
                nm_vector = sparse.kron(n_vector, m_vector, format="csc")
                pnum_projector += nm_vector * nm_vector.transpose()

        return pnum_projector

    def get_permutation_operator(self, indices, full=False):
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
        out: scipy.sparse.csc_matrix (dim, dim)
            Permutation operator for the desired sector.
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
             ), dtype=complex)

        for n in range(dim_subspace):
            perm_op_sector[n, indices[n]] = 1.0

        if full:
            for n, m in zip(n_prime_diff, m_prime_diff):
                perm_op_sector[n, m] = 1.0

            for n in n_m_prime_same:
                perm_op_sector[n, n] = 1.0

        return dim_subspace, perm_op_sector.tocsc()

    def particle_number_fock_subspace_permutation_operator(self, nelec,
                                                           full=False):
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
            Permutation operator for the desired Fock particle number sector
            "nelec".
        """
        pnum_fock = self.particle_number_fock_subspace_projector(nelec)
        pnum_fock_index = np.where(
            pnum_fock.diagonal() == 1)[0]
        return self.get_permutation_operator(pnum_fock_index, full)


class SpinSectorDecomposition(SubspaceDecomposition):
    r"""SubspaceDecomposition child class decomposes the super-fermionic
    creation and annihilation in spin sectors (\Delta N_{up},\Delta N_{do}).
    In case of spinless fermions the sectors are given by
    (\Delta N=N-N_{tilde})

    Parameters
    ----------
    parent : sf_op.SubspaceDecomposition
        Parent class
    """

    def __init__(self, nsite, spin_sector_max, target_sites=None,
                 spinless=False, tilde_conjugationrule_phase=True) -> None:
        assert spin_sector_max >= 0
        SubspaceDecomposition.__init__(
            self, nsite=nsite, spinless=spinless,
            tilde_conjugationrule_phase=tilde_conjugationrule_phase)
        self.spin_sector_max = spin_sector_max

        # TODO: default self.target_sites should in general be a list of all
        #       sites only special cases need a specific value
        if target_sites is None:
            self.target_sites = [
                int((self.fock_ops.nsite - 1) / 2)]
        else:
            self.target_sites = target_sites
        self.set_spin_subspace()

    def spin_sector_projector(self, sector):
        """Projector for given spin sector "sector" in the liouville space.


        Parameters
        ----------
        sector : int or tuple (up, do)
            Spin sector defined by the difference between particles in
            "normal" and "tilde" space for spin up and down if spinfull and
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
                         ).spin_times_site)), dtype=complex)

            for n in pnum_index:
                pnum_per_spin_projector[n, n] = 1.0
            pnum_per_spin_projector = pnum_per_spin_projector.tocsc()

            return pnum_per_spin_projector
        else:
            pnum_index = np.where(
                (self.N - self.N_tilde).diagonal(
                ) == sector)[0]

            pnum_per_spin_projector = sparse.lil_matrix(
                (4**(self.fock_ops.spin_times_site),
                    4**((self.fock_ops
                         ).spin_times_site)), dtype=complex)

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
        sector : int or tuple (up, do)
            Spin sector defined by the difference between particles in
            "normal" and "tilde" space for spin up and down is spinfull and
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

    def set_possible_spin_sectors(self):
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
            self.spin_sectors = list(map(lambda x: tuple(x),
                                         self.spin_sectors))
        else:
            self.spin_sectors = sector_range

    def set_spin_sectors_permutation_ops(self):
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

    def set_spin_sectors_fermionic_ops(self):
        """Calculate and store the fermionic operators within the accessible
        spin sectors subspace. The spin sectors are stored in
        self.spin_sectors. This reduces the dimension of the operators in
        super-fermionic space.
        """

        self.spin_sector_fermi_ops = {site: {'c': {}, 'c_tilde': {},
                                      'cdag': {}, 'cdag_tilde': {}} for site in
                                      self.target_sites}
        if not self.fock_ops.spinless:
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

    def set_spin_subspace(self):
        self.set_possible_spin_sectors()
        self.set_spin_sectors_permutation_ops()
        self.set_spin_sectors_fermionic_ops()

    def get_subspace_object(self, object, sector_left=None, sector_right=None):
        assert not ((sector_right is None) and (sector_left is None))
        if sector_left is None:
            return get_subspace_object(
                object=object,
                permutation_op_right=self.projectors[sector_right])
        elif sector_right is None:
            return get_subspace_object(
                object=object, permutation_op_left=self.projectors[sector_left]
            )
        else:
            return get_subspace_object(
                object=object,
                permutation_op_left=self.projectors[sector_left],
                permutation_op_right=self.projectors[sector_right])

    def c_sector(self, sector, site, spin=None):
        r"""Returns the "normal" space annihilation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : int or tuple
            Spin sector of interest. If spinless fermions, than use integer.
            The sector is than given by a fixed value for
            (\Delta N=N-N_{tilde}).
            If spinfull fermions, the sector is than given by a tuple with
            fixed value for (\Delta N_{up},\Delta N_{do}), e.g. (0,0).

        site : int
            site/orbital index

        spin : string, optional
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
        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if (site not in self.target_sites):
            raise IndexError('ERROR: index out of bound!')

        if self.fock_ops.spinless:
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.spin_sector_fermi_ops[site]['c'][sector]
        else:
            if spin == "up":
                return self.spin_sector_fermi_ops[site]['c']['up'][sector]
            elif spin == "do":
                return self.spin_sector_fermi_ops[site]['c']['do'][sector]
            else:
                raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def c_tilde_sector(self, sector, site, spin=None):
        r"""Returns the "tilde" space annihilation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : int or tuple
            Spin sector of interest. If spinless fermions, than use integer.
            The sector is than given by a fixed value for
            (\Delta N=N-N_{tilde}).
            If spinfull fermions, the sector is than given by a tuple with
            fixed value for (\Delta N_{up},\Delta N_{do}), e.g. (0,0).

        site : int
            site/orbital index

        spin : string, optional
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
        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if (site not in self.target_sites):
            raise IndexError('ERROR: index out of bound!')

        if self.fock_ops.spinless:
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.spin_sector_fermi_ops[site]['c_tilde'][sector]
        else:
            if spin == "up":
                return self.spin_sector_fermi_ops[site]['c_tilde']['up'][
                    sector]
            elif spin == "do":
                return self.spin_sector_fermi_ops[site]['c_tilde']['do'][
                    sector]
            else:
                raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def cdag_sector(self, sector, site, spin=None):
        r"""Returns the "normal" space creation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : int or tuple
            Spin sector of interest. If spinless fermions, than use integer.
            The sector is than given by a fixed value for
            (\Delta N=N-N_{tilde}).
            If spinfull fermions, the sector is than given by a tuple with
            fixed value for (\Delta N_{up},\Delta N_{do}), e.g. (0,0).

        site : int
            site/orbital index

        spin : string, optional
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
        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if (site not in self.target_sites):
            raise IndexError('ERROR: index out of bound!')

        if self.fock_ops.spinless:
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.spin_sector_fermi_ops[site]['cdag'][sector]
        else:
            if spin == "up":
                return self.spin_sector_fermi_ops[site]['cdag']['up'][sector]
            elif spin == "do":
                return self.spin_sector_fermi_ops[site]['cdag']['do'][sector]
            else:
                raise ValueError("ERROR: Spin can be only 'up' or 'do'!")

    def cdag_tilde_sector(self, sector, site, spin=None):
        r"""Returns the 'tilde' space creation operator at in sector
        'sector', site/orbital 'site' and with spin 'spin'

        Parameters
        ----------
        sector : int or tuple
            Spin sector of interest. If spinless fermions, than use integer.
            The sector is than given by a fixed value for
            (\Delta N=N-N_{tilde}).
            If spinfull fermions, the sector is than given by a tuple with
            fixed value for (\Delta N_{up}, \Delta N_{do}), e.g. (0,0).

        site : int
            site/orbital index

        spin : string, optional
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
        ValueError
            If spin is not 'up' or 'do' in spin 1/2 fermions
        """
        if (site not in self.target_sites):
            raise IndexError('ERROR: index out of bound!')

        if self.fock_ops.spinless:
            if spin is not None:
                print("Spinless fermions don't need the argument spin to be " +
                      "passed")
            return self.spin_sector_fermi_ops[site]['cdag_tilde'][sector]
        else:
            if spin == "up":
                return self.spin_sector_fermi_ops[site]['cdag_tilde']['up'][
                    sector]
            elif spin == "do":
                return self.spin_sector_fermi_ops[site]['cdag_tilde']['do'][
                    sector]
            else:
                raise ValueError("ERROR: Spin can be only 'up' or 'do'!")
