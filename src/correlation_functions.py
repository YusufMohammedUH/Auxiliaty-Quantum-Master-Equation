"""Here the base code for the construction of correlation functions
    will be written.
    """
# %%
import numpy as np
import src.model_hamiltonian as ham
import src.model_lindbladian as lind
import src.auxiliary_system_parameter as aux
import src.frequency_greens_function as fg
from src.dos_util import heaviside
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigs


class Correlators:
    def __init__(self, Lindbladian, spin_sector_max):
        assert spin_sector_max >= 0
        self.Lindbladian = Lindbladian
        self.spin_sector_max = spin_sector_max
        self.set_spin_sectors()

    def update_model_parameter(self, Gamma1, Gamma2, T_mat=None, U_mat=None):
        if T_mat is not None:
            self.T_mat = T_mat
        if T_mat is not None:
            self.U_mat = U_mat
        self.Gamma1 = Gamma1
        self.Gamma2 = Gamma2

    def set_lindbladian(self, sign):
        self.sign = sign
        self.Lindbladian.set_unitay_part(self.T_mat, self.U_mat)
        self.Lindbladian.set_dissipation(self.Gamma1, self.Gamma2, sign)
        self.Lindbladian.set_total_linbladian()

    def set_rho_steady_state(self):
        self.set_lindbladian(1.0)
        P_00 = (self.Lindbladian.liouville_ops
                ).spin_sector_permutation_operator((0, 0))
        L_00 = (P_00[1] * self.Lindbladian.L_tot * P_00[1].transpose()
                )[:P_00[0], :P_00[0]]

        vals, vec_r = self.Lindbladian.exact_spectral_decomposition(
            L_00, eigenvec_left=False)

        mask = np.isclose(vals, np.zeros(vals.shape))
        n_steady_state = vals[mask].shape[0]
        if n_steady_state > 1:
            raise ValueError("There are more than one stready states")
        self.rho_stready_state = vec_r[mask][0]
        self.rho_stready_state /=\
            ((self.Lindbladian.liouville_ops.left_vacuum
              ).transpose().conjugate() * P_00[1].transpose())[0, :P_00[0]]\
            * self.rho_stready_state

        # self.check_sector(self.rho_stready_state)

    def set_spin_sectors(self):
        sector_range = np.arange(-self.spin_sector_max,
                                 self.spin_sector_max + 1)
        spin_combination = np.array(
            np.meshgrid(sector_range, sector_range)).T.reshape(-1, 2)
        allowed_combinations = np.sum(np.abs(spin_combination), -1)
        allowed_combinations = allowed_combinations <= self.spin_sector_max
        self.spin_combination = spin_combination[allowed_combinations]

    def sectors_exact_decomposition(self, set_lindblad=True):
        if (self.sign != -1.0) and set_lindblad:
            self.set_lindbladian(-1.0)
        self.vals_sector = {}
        self.vec_l_sector = {}
        self.vec_r_sector = {}
        for sector in self.spin_combination:
            P_sector = (self.Lindbladian.liouville_ops
                        ).spin_sector_permutation_operator(sector)
            L_sector = (P_sector[1] * self.Lindbladian.L_tot
                        * P_sector[1].transpose()
                        )[:P_sector[0], :P_sector[0]]
            self.vals_sector[tuple(sector)], self.vec_l_sector[tuple(sector)],\
                self.vec_r_sector[tuple(sector)] = \
                self.Lindbladian.exact_spectral_decomposition(L_sector)

    def get_two_point_correlator_time(self, time, B, A, sector):
        # calculating the Lindblaian time evolution operator at time 'time'
        time_evolution_sector = \
            self.Lindbladian.time_evolution_operator(
                time, self.vals_sector[(*sector,)],
                self.vec_l_sector[(*sector,)], self.vec_r_sector[(*sector,)])
        time_evolution_minus_sector = \
            self.Lindbladian.time_evolution_operator(
                time, self.vals_sector[(*(-1 * np.array(sector)),)],
                self.vec_l_sector[(*(-1 * np.array(sector)),)],
                self.vec_r_sector[(*(-1 * np.array(sector)),)])
        # Set up Permutation operators, which permute the relevant sector
        # to the upper left of the matrix to be transformed and dropping the
        # rest.
        P_00 = (self.Lindbladian.liouville_ops
                ).spin_sector_permutation_operator((0, 0))
        P_sector = (self.Lindbladian.liouville_ops
                    ).spin_sector_permutation_operator(sector)
        P_minus_sector = (self.Lindbladian.liouville_ops
                          ).spin_sector_permutation_operator(
                              -1 * np.array(sector))
        # calculate the Operators in the given sectors
        A_sector = (P_sector[1] * A
                    * P_00[1].transpose()
                    )[:P_sector[0], :P_00[0]].todense()
        A_dagger_sector = (P_minus_sector[1] * A.transpose().conjugate()
                           * P_00[1].transpose()
                           )[:P_minus_sector[0], :P_00[0]].todense()
        B_sector = (P_00[1] * B
                    * P_sector[1].transpose()
                    )[:P_00[0], :P_sector[0]].todense()
        B_dagger_sector = (P_00[1] * B.transpose().conjugate()
                           * P_minus_sector[1].transpose()
                           )[:P_00[0], :P_minus_sector[0]].todense()
        left_vacuum_00 = ((self.Lindbladian.liouville_ops.left_vacuum
                           ).transpose().conjugate()
                          * P_00[1].transpose())[0, :P_00[0]].todense()

        G_t_plus = -1j * heaviside(time, 0) \
            * left_vacuum_00.dot(B_sector).dot(
                time_evolution_sector).dot(A_sector).dot(
                    self.rho_stready_state)
        # TODO: check formula
        G_t_minus = -1j * heaviside(time, 0) \
            * np.conj(
                left_vacuum_00.dot(B_dagger_sector).dot(
                    time_evolution_minus_sector).dot(A_dagger_sector).dot(
                    self.rho_stready_state
                ))
        return G_t_plus, G_t_minus

    def get_two_point_correlator_frequency(self, omegas, B, A, sector):
        # Set up Permutation operators, which permute the relevant sector
        # to the upper left of the matrix to be transformed and dropping the
        # rest.
        P_00 = (self.Lindbladian.liouville_ops
                ).spin_sector_permutation_operator((0, 0))
        P_sector = (self.Lindbladian.liouville_ops
                    ).spin_sector_permutation_operator(sector)
        P_minus_sector = (self.Lindbladian.liouville_ops
                          ).spin_sector_permutation_operator(
            -1 * np.array(sector))
        # calculate the Operators in the given sectors
        A_sector = (P_sector[1] * A
                    * P_00[1].transpose()
                    )[:P_sector[0], :P_00[0]].todense()
        A_dagger_sector = (P_minus_sector[1] * A.transpose().conjugate()
                           * P_00[1].transpose()
                           )[:P_minus_sector[0], :P_00[0]].todense()
        B_sector = (P_00[1] * B
                    * P_sector[1].transpose()
                    )[:P_00[0], :P_sector[0]].todense()
        B_dagger_sector = (P_00[1] * B.transpose().conjugate()
                           * P_minus_sector[1].transpose()
                           )[:P_00[0], :P_minus_sector[0]].todense()
        left_vacuum_00 = ((self.Lindbladian.liouville_ops.left_vacuum
                           ).transpose().conjugate()
                          * P_00[1].transpose())[0, :P_00[0]].todense()

        G_omega_plus = np.zeros(omegas.shape, dtype=complex)
        G_omega_minus = np.zeros(omegas.shape, dtype=complex)
        for n, omega in enumerate(omegas):
            for m in range(self.vals_sector[(*sector,)].shape[0]):
                G_omega_plus[n] += left_vacuum_00.dot(B_sector).dot(
                    self.vec_r_sector[(*sector,)][m]).dot(
                    self.vec_l_sector[(*sector,)][m]).dot(A_sector).dot(
                    self.rho_stready_state) * (
                    1.0 / (omega
                           - 1j * self.vals_sector[(*sector,)][m]))
            # TODO: check formula
            for m in range(self.vals_sector[(
                    *(-1 * np.array(sector)),)].shape[0]):
                G_omega_minus[n] += np.conj(left_vacuum_00.dot(
                    B_dagger_sector).dot(
                    self.vec_r_sector[(*(-1 * np.array(sector)),)][m]).dot(
                    self.vec_l_sector[(*(-1 * np.array(sector)),)][m]).dot(
                        A_dagger_sector).dot(self.rho_stready_state)) * (
                            1.0 / (omega - 1j * np.conj(
                                self.vals_sector[(
                                    *(-1 * np.array(sector)),)][m])))
        return G_omega_plus, G_omega_minus


# %%
if __name__ == "__main__":
    Nb = 1
    nsite = 2 * Nb + 1
    ws = np.linspace(-5, 5, 201)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.2 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])
    Us = np.zeros(nsite)
    Us[Nb] = 2.  # 4.
    # initializing auxiliary system and E, Gamma1 and Gamma2 for a
    # particle-hole symmetric system
    sys = aux.AuxiliarySystem(Nb, ws)
    sys.set_ph_symmetric_aux(es, ts, gamma)

    green = fg.FrequencyGreen(sys.ws)
    green.set_green_from_auxiliary(sys)

    hyb_aux = green.get_self_enerqy()
    # %%

    spinless = False
    # Initializing Lindblad class
    L = lind.Lindbladian(nsite, spinless,
                         Dissipator=lind.Dissipator_thermal_bath)

    # Setting unitary part of Lindbladian
    T_mat = sys.E
    T_mat[Nb, Nb] = -Us[Nb] / 2.0
    L.set_unitay_part(T_mat=T_mat, U_mat=Us)

    # Setting dissipative part of Lindbladian
    L.set_dissipation(sys.Gamma1, sys.Gamma2)

    # Setting total Lindbladian
    L.set_total_linbladian()

    corr = Correlators(L, 1)
    corr.update_model_parameter(sys.Gamma1, sys.Gamma2, T_mat, Us)
    corr.set_rho_steady_state()
    corr.sectors_exact_decomposition()

    # %%
    G_plus, G_minus = corr.get_two_point_correlator_frequency(
        ws, corr.Lindbladian.liouville_ops.c(1, "up"),
        corr.Lindbladian.liouville_ops.cdag(1, "up"), (1, 0))
    # %%
    G_R = G_plus + G_minus
    G_K = G_plus + np.conj(G_minus) - np.conj(G_plus + np.conj(G_minus))
    green2 = fg.FrequencyGreen(sys.ws, retarded=G_R, keldysh=G_K)
    sigma = green2.get_self_enerqy() - hyb_aux
    # %%
    plt.figure()
    plt.plot(sys.ws, green.retarded.imag)
    plt.plot(sys.ws, G_R.imag)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$ImG^R_{aux,0}(\omega)$",
                r"$ImG^R_{aux}(\omega)$"])
    plt.show()

    plt.figure()
    plt.plot(sys.ws, green.keldysh.imag)
    # plt.plot(sys.ws, green.keldysh.real)
    plt.plot(sys.ws, G_K.imag)
    # plt.plot(sys.ws, G_K.real)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$ImG^K_{aux,0}(\omega)$",
                r"$ImG^K_{aux}(\omega)$"])
    plt.show()

    plt.figure()
    plt.plot(sys.ws, hyb_aux.retarded.imag)
    # plt.plot(sys.ws, hyb_aux.retarded.real)
    plt.plot(sys.ws, sigma.retarded.imag)
    # plt.plot(sys.ws, sigma.retarded.real)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$Im\Delta^R_{aux}(\omega)$",
                r"$Im\Sigma^R_{aux}(\omega)$"])
    plt.show()
# %%
# %%
