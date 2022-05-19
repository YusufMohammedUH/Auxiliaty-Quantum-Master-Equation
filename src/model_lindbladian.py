# %%
from scipy import sparse
import src.define_super_fermionic_operators as sop
import src.super_fermionic_subspace as sop_sub
import src.model_hamiltonian as ham
# TODO: Write tests
#       Here it's probably best to reproduce papers
#      [x] Spinless fermions without interaction coupled to
#          Dissipator_thermal_radiation_mode
#      [ ] Spinless fermions with interaction coupled to
#          Dissipator_thermal_radiation_mode

#      [ ] Spinfull fermions without interaction coupled to
#          Dissipator_thermal_radiation_mode
#      [ ] Spinfull fermions with interaction coupled to
#          Dissipator_thermal_radiation_mode

#      [ ] Spinless fermions without interaction coupled to
#          Dissipator_thermal_bath
#      [ ] Spinless fermions with interaction coupled to
#          Dissipator_thermal_bath

#      [ ] Spinfull fermions without interaction coupled to
#          Dissipator_thermal_bath
#      [ ] Spinfull fermions with interaction coupled to
#          Dissipator_thermal_bath


def Dissipator_thermal_bath(Gamma1, Gamma2, super_fermi_ops, sign=1,
                            tilde_conjugationrule_phase=False):
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

    super_fermi_ops : SuperFermionicOperators
        Contains the fermionic creation and annihilation operators Liouville
        space and converts Fock space operators in to Liouville space operators

    sign : float, 1 or -1
        The sign in the dissipator, due to the fermionic commutation relation.
        This is -1 if the number of operators in the dissipator and the
        fermionic operator of interest are odd if the tilde conjugation rule
        is applied without a complex phase, 1 otherwise.

    tilde_conjugationrule_phase: bool, optional
            If True it is assumed that the tilde conjugation rule is applied
            with an imaginary phase for the left vacuum state. The tilde
            operators pick up a additional complex phase. If False,
            the phase is set to one, by default False

    Returns
    -------
    (L_Gamma1,L_Gamma2): scipy.sparse.csc_matrix (dim,dim)
        Tuple of the Liouville dissipator, with L_Gamma1, describing the
        removal of electrons, and L_Gamma2, describing the injection of
        electrons.
    """
    nsite = super_fermi_ops.fock_ops.nsite
    spin_times_site = super_fermi_ops.fock_ops.spin_times_site
    L_Gamma1 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    L_Gamma2 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    if super_fermi_ops.fock_ops.spinless:
        spins = [None]
    else:
        spins = ["up", "do"]

    if super_fermi_ops.tilde_conjugationrule_phase:
        print("Dissipative Lindbladian is set with tilde conjugation rule.")
        for ii in range(nsite):
            for jj in range(nsite):
                for spin in spins:
                    if Gamma1[ii, jj] != 0:
                        L_Gamma1 += (2. * super_fermi_ops.tilde_operator_sign[
                            "c_tilde"] * Gamma1[ii, jj]
                            * super_fermi_ops.c(jj, spin)
                            * super_fermi_ops.c_tilde(ii, spin)
                            - Gamma1[ii, jj]
                            * (super_fermi_ops.cdag(ii, spin)
                               * super_fermi_ops.c(jj, spin)
                               + super_fermi_ops.cdag_tilde(jj, spin)
                               * super_fermi_ops.c_tilde(ii, spin))
                        )
                    if Gamma2[ii, jj] != 0:
                        L_Gamma2 += (2. * super_fermi_ops.tilde_operator_sign[
                            "cdag_tilde"] * Gamma2[ii, jj]
                            * super_fermi_ops.cdag(ii, spin)
                            * super_fermi_ops.cdag_tilde(jj, spin)
                            - Gamma2[ii, jj]
                            * (super_fermi_ops.c(jj, spin)
                               * super_fermi_ops.cdag(ii, spin)
                               + super_fermi_ops.c_tilde(ii, spin)
                               * super_fermi_ops.cdag_tilde(jj, spin))
                        )
        return L_Gamma1, L_Gamma2

    for ii in range(nsite):
        for jj in range(nsite):
            for spin in spins:
                if Gamma1[ii, jj] != 0:
                    L_Gamma1 += (2. * super_fermi_ops.tilde_operator_sign[
                        "c_tilde"]
                        * Gamma1[ii, jj]
                        * super_fermi_ops.c(jj, spin)
                        * super_fermi_ops.c_tilde(ii, spin)
                        - Gamma1[ii, jj]
                        * (super_fermi_ops.cdag(ii, spin)
                           * super_fermi_ops.c(jj, spin)
                           + super_fermi_ops.cdag_tilde(jj, spin)
                           * super_fermi_ops.c_tilde(ii, spin))
                    )
                if Gamma2[ii, jj] != 0:
                    L_Gamma2 += (2. * super_fermi_ops.tilde_operator_sign[
                        "cdag_tilde"]
                        * Gamma2[ii, jj]
                        * super_fermi_ops.cdag(ii, spin)
                        * super_fermi_ops.cdag_tilde(jj, spin)
                        - Gamma2[ii, jj]
                        * (super_fermi_ops.c(jj, spin)
                           * super_fermi_ops.cdag(ii, spin)
                           + super_fermi_ops.c_tilde(ii, spin)
                           * super_fermi_ops.cdag_tilde(jj, spin))
                    )
    return L_Gamma1, L_Gamma2


def Dissipator_thermal_radiation_mode(Gamma1, Gamma2, super_fermi_ops, sign=1,
                                      tilde_conjugationrule_phase=False):
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

    super_fermi_ops : SuperFermionicOperators
        Contains the fermionic creation and annihilation operators Liouville
        space and converts Fock space operators in to Liouville space operators

    Returns
    -------
    (L_Gamma1,L_Gamma2): scipy.sparse.csc_matrix (dim,dim)
        Tuple of the Liouville dissipator, with L_Gamma1, describing the
        removal of electrons, and L_Gamma2, describing the injection of
        electrons.
    """
    nsite = super_fermi_ops.fock_ops.nsite
    spin_times_site = super_fermi_ops.fock_ops.spin_times_site
    L_Gamma1 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    L_Gamma2 = sparse.csc_matrix(
        (4**(spin_times_site), 4**(spin_times_site)), dtype=complex)
    if super_fermi_ops.fock_ops.spinless:
        spins = [None]
    else:
        spins = ["up", "do"]

    for ii in range(nsite):
        for jj in range(nsite):
            for spin in spins:
                if jj > ii:
                    if Gamma1[ii, jj] != 0:
                        L_Gamma1 += (Gamma1[ii, jj]
                                     * super_fermi_ops.cdag(ii, spin)
                                     * super_fermi_ops.c(jj, spin)
                                     * super_fermi_ops.cdag_tilde(ii, spin)
                                     * super_fermi_ops.c_tilde(jj, spin)
                                     - 0.5 * Gamma1[ii, jj]
                                     * (super_fermi_ops.cdag(jj, spin)
                                        * super_fermi_ops.c(ii, spin)
                                        * super_fermi_ops.cdag(ii, spin)
                                        * super_fermi_ops.c(jj, spin)
                                        + super_fermi_ops.cdag_tilde(jj, spin)
                                        * super_fermi_ops.c_tilde(ii,
                                        spin)
                                        * super_fermi_ops.cdag_tilde(ii, spin)
                                        * super_fermi_ops.c_tilde(jj,
                                        spin))
                                     )

                    if Gamma2[ii, jj] != 0:
                        L_Gamma2 += (Gamma2[ii, jj]
                                     * super_fermi_ops.cdag(jj, spin)
                                     * super_fermi_ops.c(ii, spin)
                                     * super_fermi_ops.cdag_tilde(jj, spin)
                                     * super_fermi_ops.c_tilde(ii, spin)
                                     - 0.5 * Gamma2[ii, jj]
                                     * (super_fermi_ops.cdag(ii, spin)
                                        * super_fermi_ops.c(jj, spin)
                                        * super_fermi_ops.cdag(jj, spin)
                                        * super_fermi_ops.c(ii, spin)
                                        + super_fermi_ops.cdag_tilde(ii, spin)
                                        * super_fermi_ops.c_tilde(jj,
                                        spin)
                                        * super_fermi_ops.cdag_tilde(jj, spin)
                                        * super_fermi_ops.c_tilde(ii,
                                        spin))
                                     )
    return L_Gamma1, L_Gamma2


class Lindbladian:
    def __init__(self, super_fermi_ops,
                 Hamiltonian=ham.hubbard_hamiltonian,
                 Dissipator=Dissipator_thermal_bath) -> None:
        """Class for setting up a Lindbladian

        Parameters
        ----------
        nsite : int
            number of sites/ orbitals of the fermionic system

        super_fermi_ops: src.SuperFermionicOperators type
            This object defines the super-fermionic space of the
            lindbladian

        spinless : bool, optional
            Indicates if the fermions are spinless, by default False

        tilde_conjugationrule_phase: bool, optional
            If True a the tilde conjugation rule is applied with an imaginary
            phase for the tilde operators and the left vacuum state. If False,
            the phase is set to one, by default False

        Hamiltonian : function returning a sparse matrix, optional
            Hamiltonian of the reduced system, by default
            ham.hubbard_hamiltonian

        Dissipator : function returning a tuple of sparse matrices, optional
            Function describing the dissipation/coupling between the reduced
            system and the surrounding bath, by default Dissipator_thermal_bath

        Attributes
        ----------
        super_fermi_ops:  SuperFermionicOperators
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
        self.super_fermi_ops = super_fermi_ops
        self.Hamiltonian = Hamiltonian
        self.Dissipator = Dissipator

        self.T_mat = None
        self.U_mat = None
        self.Gamma1 = None
        self.Gamma2 = None

        self.L_unitary = None
        self.L_Gamma1 = None
        self.L_Gamma2 = None
        self.L_tot = None

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
            T_mat, U_mat, eop=self.super_fermi_ops.fock_ops,
            spinless=self.super_fermi_ops.fock_ops.spinless)

        self.L_unitary = -1.j * \
            (self.super_fermi_ops.get_super_fermionic_operator(Hamil_Fock)
             - self.super_fermi_ops.get_super_fermionic_tilde_operator(Hamil_Fock)
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
        nsite = self.super_fermi_ops.fock_ops.nsite
        if (Gamma1.shape != Gamma2.shape) and Gamma1.shape != (nsite, nsite):
            raise ValueError("ERROR: Wrong shape of Gamma matrices. They have"
                             + F" to be {(nsite,nsite)}.")
        if self.super_fermi_ops.tilde_conjugationrule_phase:
            self.L_Gamma1, self.L_Gamma2 = self.Dissipator(
                Gamma1, Gamma2, self.super_fermi_ops,
                tilde_conjugationrule_phase=self.super_fermi_ops.tilde_conjugationrule_phase)
        else:
            self.L_Gamma1, self.L_Gamma2 = self.Dissipator(
                Gamma1, Gamma2, self.super_fermi_ops, sign=sign)

    def set_total_linbladian(self):
        """Set the total Lindbladian"""
        self.L_tot = self.L_unitary + self.L_Gamma1 + \
            self.L_Gamma2

    def update(self, T_mat=None, U_mat=None, Gamma1=None,
               Gamma2=None, sign=None):
        """Update the model parameter and recalculate the Lindbladian.

        _extended_summary_

        Parameters
        ----------
        T_mat : numpy.ndarray (dim,dim), optional
            Hopping matrix, by default None

        U_mat : numpy.ndarray (dim,) or (dim,dim), optional
            On-site interaction strength. It can differ at each site,
            by default None. By default None

        Gamma1 : numpy.ndarray (dim,dim), optional
            2D array with the coupling to bath 1, describing the removal of
            electrons. By default None

        Gamma2 : numpy.ndarray (dim,dim), optional
            2D array with the coupling to bath 2, describing the injection of
            electrons. By default None

        sign : int (-1 or 1), optional
            -1 if the Lindbladian describes the time evolution of a single
            fermionic operator and with a dissipator describing a single
            electron dissipation, 1 otherwise. By default None

        Raises
        ------
        Warning
            T_mat and U_mat are expected to be simultaneously passed or None.
        Warning
            Gamma1 and Gamma2 are expected to be simultaneously passed or None.
        """

        if (T_mat is not None) != (U_mat is not None):
            raise Warning("WARNING: T_mat and U_mat are not updated" +
                          " simultaneously. This should be done if the" +
                          " unitary part is to be updated.")

        if (T_mat is not None) and (T_mat != self.T_mat):
            self.T_mat = T_mat
        if (U_mat is not None) and (U_mat != self.U_mat):
            self.U_mat = U_mat
        if (T_mat is not None) or (U_mat is not None):
            self.set_unitay_part(self.T_mat, self.U_mat)

        if (Gamma1 is not None) != (Gamma2 is not None):
            raise Warning("WARNING: Gamma1 and Gamma2 are not updated" +
                          " simultaneously. This should be done if the" +
                          " dissipative part is to be updated.")
        if Gamma1 is not None:
            self.Gamma1 = Gamma1
        if Gamma2 is not None:
            self.Gamma2 = Gamma2
        if (Gamma1 is not None) or (Gamma2 is not None) or (sign is not None):
            self.set_dissipation(self.Gamma1, self.Gamma2, sign=sign)

        if ((T_mat is not None) or (U_mat is not None) or
            (Gamma1 is not None) or (Gamma2 is not None) or
                (sign is not None)):
            self.set_total_linbladian()
