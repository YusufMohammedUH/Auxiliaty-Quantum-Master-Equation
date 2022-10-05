""" Exact decomposition solver
This file defines the EDSolver class, which can be used to calculate
correlation functions.
    """
# %%
from typing import Tuple, Union
import numpy as np
import src.liouville_solvers.ed_solver_get_correlator_jit as corr
import src.liouville_solvers.ed_solver_precalculate_jit as precalc
import src.liouville_solvers.exact_decomposition as ed
import src.super_fermionic_space.model_lindbladian as lind
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.greens_function.frequency_greens_function as fg
import src.auxiliary_mapping.auxiliary_system_parameter as aux


class EDSolver:
    """EDSolver(self, Lindbladian: lind.Lindbladian,
                 spin_sectors: Union[int, None] = None) -> None
    Exact diagonalization solver for calculating correlation functions
    from supplied Lindbladian and super-fermionic space

     Parameters
    ----------
    Lindbladian : lind.Lindbladian
        Lindbladian object, containing the Lindbladian and the corresponding
        super-fermionic supspace class.
    spin_sectors : Union[int, None], optional
        All reachable/desired spin sectors, by default None

    """

    def __init__(self, Lindbladian: lind.Lindbladian,
                 spin_sectors: Union[int, None] = None) -> None:

        if spin_sectors is None:
            spin_sectors = Lindbladian.super_fermi_ops.spin_sectors
        else:
            for sector in spin_sectors:
                assert sector in Lindbladian.super_fermi_ops.spin_sectors
        self.spin_sectors = spin_sectors
        self.spinless = Lindbladian.super_fermi_ops.fock_ops.spinless

        self.vals_sector = {}
        self.vec_l_sector = {}
        self.vec_r_sector = {}
        self.rho_stready_state = None
        self.left_vacuum = None
        self.precalc_correlator = None
        self.e_cut_off = None

    def prepare(self, correlators: dict = None) -> None:
        """Set precalculated expectation values keys.

        Parameters
        ----------
        correlators : dict, optional
            Dictionary in which the correlators are saved, by default None
        """
        if correlators is None:
            self.precalc_correlator = {n: {} for
                                       n in self.precalc_correlator.keys()}
        else:
            self.precalc_correlator = {n: {} for
                                       n in correlators}

    def sector_exact_decompsition(self, Lindbladian: lind.Lindbladian) -> None:
        """Exactly decompose the Lindbladian within the relevant spin
        sectors. The eigenvectors and eigenvalues are saved as object
        attributes.

        Parameters
        ----------
        Lindbladian : lind.Lindbladian
            Lindbladian class for a given system
        """
        assert \
            self.spinless == Lindbladian.super_fermi_ops.fock_ops.spinless, (
                "ERROR: Type of fermions are different")

        for sector in self.spin_sectors:
            if not Lindbladian.super_fermi_ops.fock_ops.spinless:
                sector = tuple(sector)
            L_sector = Lindbladian.super_fermi_ops.get_subspace_object(
                Lindbladian.L_tot, sector, sector)
            self.vals_sector[sector], self.vec_l_sector[sector],\
                self.vec_r_sector[sector] = \
                ed.exact_spectral_decomposition(L_sector.todense())

    def get_rho_steady_state(self, Lindbladian: lind.Lindbladian = None
                             ) -> None:
        """Calculate the steady state density of state in the spin sector
        (0,0). It has to be unique.

        Parameters
        ----------
        Lindbladian : lind.Lindbladian
            Lindbladian class for a given system

        Raises
        ------
        ValueError
            If there are more then one steady-steate density of states
        """
        if self.spinless:
            sector0 = 0
        else:
            sector0 = (0, 0)
        if Lindbladian is None:
            vals = self.vals_sector[sector0]
            vec_r = self.vec_r_sector[sector0]
        else:
            L_00 = Lindbladian.super_fermi_ops.get_subspace_object(
                Lindbladian.L_tot, sector0, sector0)

            vals, _, vec_r = ed.exact_spectral_decomposition(
                L_00.todense())

        mask = np.isclose(vals, np.zeros(vals.shape))
        vals_close_zero = vals[mask]
        n_steady_state = vals_close_zero.shape[0]
        if n_steady_state == 0:
            raise ValueError("No steady state density of state.")
        if n_steady_state > 1:
            vals_close_zero_sorted = sorted(
                vals_close_zero, key=(lambda x: np.abs(x)))
            vals_renormalized = [e / np.abs(vals_close_zero_sorted[0])
                                 for e in vals_close_zero_sorted]
            mask2 = vals_close_zero == vals_close_zero_sorted[0]
            if np.abs(vals_renormalized[1]) < 100:
                print("eigen values")
                raise ValueError(
                    "There are more than one stready states values")
            e_cut_off_tmp = np.abs(vals_close_zero_sorted[0] * 100)
            e_cut_off_tmp = np.round(e_cut_off_tmp, -int(
                np.floor(np.log10(abs(e_cut_off_tmp)))))
            self.rho_stready_state = vec_r[mask][mask2][0]
        else:
            e_cut_off_tmp = np.abs(vals_close_zero[0] * 100)
            e_cut_off_tmp = np.round(e_cut_off_tmp, -int(
                np.floor(np.log10(abs(e_cut_off_tmp)))))
            self.rho_stready_state = vec_r[mask][0]
        self.e_cut_off = e_cut_off_tmp
        self.left_vacuum = \
            Lindbladian.super_fermi_ops.get_subspace_object(
                (Lindbladian.super_fermi_ops.left_vacuum
                 ).transpose().conjugate(),
                sector_right=sector0).todense()

        self.rho_stready_state /= self.left_vacuum * self.rho_stready_state

    def get_expectation_value(self, operator_sector00: np.ndarray) -> complex:
        """Get expectation value of an operator in the ((0,0),(0,0)) sector.

        Parameters
        ----------
        operator_sector00 : np.ndarray
            operator in the ((0,0),(0,0)) sector

        Returns
        -------
        out: complex
            expectation value
        """
        return self.left_vacuum.dot(operator_sector00).dot(
            self.rho_stready_state)

    def update(self, Lindbladian: lind.Lindbladian) -> None:
        """Update the exact decomposition.

        Parameters
        ----------
        Lindbladian : lind.Lindbladian
            Lindbladian class for a given system
        """
        self.prepare()
        self.sector_exact_decompsition(Lindbladian=Lindbladian)

    def precalculate_expectation_value(self, Lindbladian: lind.Lindbladian,
                                       sites: Tuple, operator_keys: Tuple,
                                       shape: Tuple) -> None:
        """Calculate the expectation value of a given set of operators with
        regards to the steady state density of states and at time zero. It is
        calculated by insertion of the Lindbladian eigenbasis and saved
        separately, e.g.
        <I|A|R_n><L_n|B|R_m><L_n|C|rho>=:D_nm  with n in {0,...,N} and m in
        {0,...,M}
        The operators A,B,C can be a collection fermionic operators.

        This can later be used to determine the time/frequency dependent
        expectation values.

        Parameters
        ----------
        operator_keys : tuple
            keys of the operators containing the sectors, the spin, the
        shape : tuple of ints
            containing the dimension of the expectation value
        n_fop : int
            number of fermionic creation and annihilation operators
        n_time : int
            number of operators A,B,C
        """
        # precalculate the needed terms for calculating the corresponding
        # correlation function
        n = len(operator_keys)
        assert n < 5

        if operator_keys not in self.precalc_correlator[n].keys():
            spin_sector_fermi_ops = []
            vec_r_sector = []
            vec_l_sector = []
            for site, op_key in zip(sites, operator_keys):
                tmp = None
                operator, spin_or_channel, sector = op_key

                if operator == 'cdag':
                    tmp = ((Lindbladian.super_fermi_ops
                            ).cdag_sector(sector, site, spin_or_channel
                                          ).todense() *
                           (Lindbladian.super_fermi_ops
                            ).tilde_operator_sign[operator])
                elif operator == 'c':
                    tmp = ((Lindbladian.super_fermi_ops
                            ).c_sector(sector, site, spin_or_channel
                                       ).todense() *
                           (Lindbladian.super_fermi_ops
                            ).tilde_operator_sign[operator])
                elif operator == 'cdag_tilde':
                    tmp = ((Lindbladian.super_fermi_ops
                            ).cdag_tilde_sector(sector, site, spin_or_channel
                                                ).todense() *
                           (Lindbladian.super_fermi_ops
                            ).tilde_operator_sign[operator])
                elif operator == 'c_tilde':
                    tmp = ((Lindbladian.super_fermi_ops
                            ).c_tilde_sector(sector, site, spin_or_channel
                                             ).todense() *
                           (Lindbladian.super_fermi_ops
                            ).tilde_operator_sign[operator])
                elif operator == 'n_channel':
                    tmp = (Lindbladian.super_fermi_ops
                           ).n_channel_sector(sector, site, spin_or_channel
                                              ).todense()
                elif operator == 'n_channel_tilde':
                    tmp = (Lindbladian.super_fermi_ops
                           ).n_channel_tilde_sector(sector, site,
                                                    spin_or_channel).todense()

                spin_sector_fermi_ops.append(tmp)

            spin_sector_fermi_ops = tuple(spin_sector_fermi_ops)
            vec_r_sector = tuple(
                [self.vec_r_sector[
                    op_key[2][0]] for op_key in
                    operator_keys[1:]])
            vec_l_sector = tuple(
                [self.vec_l_sector[
                    op_key[2][0]] for op_key in
                    operator_keys[1:]])

            if n == 2:
                self.precalc_correlator[n][operator_keys] = \
                    precalc.precalculate_two_point_correlator(
                    shape, self.left_vacuum,
                    spin_sector_fermi_ops,
                    vec_r_sector,
                    vec_l_sector,
                    self.rho_stready_state, n)
            elif n == 3:
                self.precalc_correlator[n][operator_keys] = \
                    precalc.precalculate_three_point_correlator(
                    shape, self.left_vacuum,
                    spin_sector_fermi_ops,
                    vec_r_sector,
                    vec_l_sector,
                    self.rho_stready_state, n)
            elif n == 4:
                self.precalc_correlator[n][operator_keys] = \
                    precalc.precalculate_four_point_correlator(
                    shape, self.left_vacuum,
                    spin_sector_fermi_ops,
                    vec_r_sector,
                    vec_l_sector,
                    self.rho_stready_state, n)

    def get_correlator(self, Lindbladian: lind.Lindbladian, freq: np.ndarray,
                       component: Tuple, sites: Tuple, operator_keys: Tuple,
                       permutation_sign: Tuple, prefactor: Tuple[complex, None] = -1 + 0j
                       ) -> Union[Tuple[np.ndarray, np.ndarray],
                                  np.ndarray]:
        """Calculate correlation function.


        Parameters
        ----------
        Lindbladian : lind.Lindbladian
            Lindbladian class containing the lindbladian and the
            super-fermionic space

        freq : np.ndarray
            1D frequency grid

        component : Tuple
            Component,e.g. (+--) -> (1,0,0)

        sites : Tuple
            Site index of the operators passed as keys in operator_keys

        operator_keys : Tuple
            containing the spin sector, the operator name and the spin
            component.

        permutation_sign : Tuple
            Sign picked up by permuting the operators
        prefactor : complex, optional
            Prefactor of the correlator, e.g. -1j for a single particle greens
            function, by default -1+0j

        Returns
        -------
        out : Unioin[Tuple[np.ndarray,np.ndarray],np.ndarray[np.ndarray]]
            Return the green's function component.
        """
        n = len(sites)

        # Save range total number of eigenvalues in each sector
        # serves as upper int for iterating over the eigenvalues in jited
        # function
        tensor_shapes = [tuple([self.vals_sector[op_key[2][1]].shape[0]
                                for op_key in op_keys[:-1]]) for op_keys
                         in operator_keys]
        # precalculate expectation values at t=0. Reduces computation.
        precalc_correlators = []
        for op_key, tensor_shape in zip(operator_keys, tensor_shapes):
            self.precalculate_expectation_value(Lindbladian,
                                                sites, op_key,
                                                tensor_shape)
            precalc_correlators.append(self.precalc_correlator[n][op_key])
        precalc_correlators = tuple(precalc_correlators)
        # calculate single particle green's function
        if n == 2:
            # Save eigenvalues in each sector used in jited function
            vals_sectors = tuple([tuple([self.vals_sector[op_key[2][0]]
                                         for op_key in op_keys[1:]])[0]
                                  for op_keys in operator_keys])

            green_component_plus = np.zeros(freq.shape, dtype=np.complex128)
            green_component_minus = np.zeros(freq.shape, dtype=np.complex128)
            # Calculate single particle green's function or susceptibility
            # component
            corr.get_two_point_correlator_frequency(
                green_component_plus=green_component_plus,
                green_component_minus=green_component_minus, freq=freq,
                precalc_correlators=precalc_correlators,
                vals_sectors=vals_sectors, tensor_shapes=tensor_shapes,
                permutation_sign=permutation_sign, prefactor=prefactor,
                e_cut_off=self.e_cut_off)

            return green_component_plus, green_component_minus
        # calculate three point correlation function
        elif n == 3:
            # Save eigenvalues in each sector used in jited function
            vals_sectors = [tuple([self.vals_sector[op_key[2][0]]
                                   for op_key in op_keys[1:]])
                            for op_keys in operator_keys]
            # Calculate correlator component
            green_component = np.zeros((freq.shape[0], freq.shape[0]),
                                       dtype=np.complex128)
            if component == (0, 0, 0):
                corr.get_three_point_correlator_frequency_mmm(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)

            elif component == (1, 0, 0):
                corr.get_three_point_correlator_frequency_pmm(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)

            elif component == (0, 1, 0):
                corr.get_three_point_correlator_frequency_mpm(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)
            elif component == (0, 0, 1):
                corr.get_three_point_correlator_frequency_mmp(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)

            elif component == (1, 1, 0):
                corr.get_three_point_correlator_frequency_ppm(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)

            elif component == (1, 0, 1):
                corr.get_three_point_correlator_frequency_pmp(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)

            elif component == (0, 1, 1):
                corr.get_three_point_correlator_frequency_mpp(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)

            elif component == (1, 1, 1):
                corr.get_three_point_correlator_frequency_ppp(
                    green_component, freq, precalc_correlators, vals_sectors,
                    tensor_shapes, permutation_sign, prefactor, self.e_cut_off)
            return green_component


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # Set parameters
    ws = np.linspace(-10, 10, 200)

    Nb = 1
    nsite = 2 * Nb + 1
    U_imp = 0.0
    es = np.array([1])
    ts = np.array([0.5])
    Us = np.zeros(nsite)
    Us[Nb] = U_imp
    gamma = np.array([0.2 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])

    # Initializing Lindblad class
    spin_sector_max = 1
    spinless = True
    tilde_conjugationrule_phase = True

    super_fermi_ops = sf_sub.SpinSectorDecomposition(
        nsite, spin_sector_max, spinless=spinless,
        tilde_conjugationrule_phase=tilde_conjugationrule_phase)

    L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)

    # Initializing auxiliary system and E, Gamma1 and Gamma2 for a
    # particle-hole symmetric system
    sys = aux.AuxiliarySystem(Nb, ws)
    sys.set_ph_symmetric_aux(es, ts, gamma)

    green = fg.FrequencyGreen(sys.ws)
    green.set_green_from_auxiliary(sys)
    hyb_aux = green.get_self_enerqy()

    # Setting unitary part of Lindbladian
    T_mat = sys.E
    T_mat[Nb, Nb] = -Us[Nb] / 2.0

    # Setting dissipative part of Lindbladian
    print("after setting dissipator")
    L.update(T_mat=T_mat, U_mat=Us, Gamma1=sys.Gamma1,
             Gamma2=sys.Gamma2)
    ed_s = EDSolver(L)
    correlators_ = {n: {} for n in range(2, 5)}
    ed_s.prepare(correlators_)
    ed_s.update(L)

# %%
