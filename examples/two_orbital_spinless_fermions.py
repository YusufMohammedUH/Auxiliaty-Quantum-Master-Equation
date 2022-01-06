"""Time evolution of the density operator of a two site spinless fermionic
    system. Starting point is one electron in the upper level. The dissipator
    of the master equation describes spontaneous emission processes as well
    as thermally induced emission and absorption processes.
    """
# %%
import numpy as np
import src.model_hamiltonian as ham
import src.model_lindbladian as lind
import matplotlib.pyplot as plt

# ########################### Unitary propagation #############################
# Local Hamiltonian parameters of two site Problem
nsite = 2
ts = -1.0 * np.ones(nsite - 1)
U = np.zeros(nsite)
es = np.array([0., 1.])

spinless = True
# Initializing Lindblad class
L = lind.Lindbladian(nsite, spinless,
                     Dissipator=lind.Dissipator_thermal_radiation_mode)

# Setting hopping matrix
Tmat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(nsite, es, ts)

# Setting unitary part of Lindbladian
L.set_unitay_part(T_mat=Tmat, U_mat=U)

# Setting dissipative part of Lindbladian
G1 = np.zeros((nsite, nsite))
G2 = np.zeros((nsite, nsite))
L.set_dissipation(G1, G2)

# Setting total Lindbladian
L.set_total_linbladian()

# Using exact diagonalization to obtain eigenvalues and eigenvectors
L.exact_spectral_decomposition()

# defining initial desity operator with one electron in upper level
dim = int(np.sqrt(L.L_tot.shape[0]))
rho0 = np.zeros((dim, dim), dtype=complex)
rho0[2, 2] = 1

# Time evolution parameters
t_min = 0
t_max = 5
N = 1001
times = np.linspace(t_min, t_max, N)

rho = L.time_propagation_all_times_exact_diagonalization(
    times, rho0.reshape((dim**2, 1)))

plt.title("Unitary Population Dynamics")
plt.xlabel("t")
plt.plot(times, rho[10].real)
plt.plot(times, rho[5].real)

plt.show()

# ########################## Spontaneous emission ###########################
n = 0
ts = np.zeros(nsite - 1)

# Setting hopping matrix
Tmat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(nsite, es, ts)

# Setting unitary part of Lindbladian
L.set_unitay_part(T_mat=Tmat, U_mat=U)

# Setting dissipative part of Lindbladian
G1 = (0.2 * (n + 1)) * np.rot90(np.eye(nsite))
G2 = (0.2 * (n)) * np.rot90(np.eye(nsite))
L.set_dissipation(G1, G2)

# Setting total Lindbladian
L.set_total_linbladian()

# Using exact diagonalization to obtain eigenvalues and eigenvectors
L.exact_spectral_decomposition()

t_min = 0
t_max = 10
N = 1001
times = np.linspace(t_min, t_max, N)

rho = L.time_propagation_all_times_exact_diagonalization(
    times, rho0.reshape((dim**2, 1)))

plt.title("Spontaneous Emission")
plt.xlabel("t")
plt.plot(times, rho[10].real)
plt.plot(times, rho[5].real)

plt.show()

# ##################### Unitary And Incoherent Dynamics #######################

ts = -1.0 * np.ones(nsite - 1)

# Setting hopping matrix
Tmat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(nsite, es, ts)

# Setting unitary part of Lindbladian
L.set_unitay_part(T_mat=Tmat, U_mat=U)

# Setting total Lindbladian
L.set_total_linbladian()

# Using exact diagonalization to obtain eigenvalues and eigenvectors
L.exact_spectral_decomposition()

rho = L.time_propagation_all_times_exact_diagonalization(
    times, rho0.reshape((dim**2, 1)))

plt.title("Incoherent Population Dynamics")
plt.xlabel("t")
plt.plot(times, rho[10].real)
plt.plot(times, rho[5].real)

plt.show()
# %%
