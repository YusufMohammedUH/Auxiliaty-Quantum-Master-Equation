"""Example for auxliliary quantum master equation with DMFT
    self-consistency.
"""
import numpy as np
import src.model_lindbladian as lind
import src.frequency_greens_function as fg
import src.dos_util as du
import src.optimization_auxiliary_hybridization as opt
import src.correlation_functions as corr
import matplotlib.pyplot as plt

# ############################### Parameters ##################################
# Parameters of the system
e0 = 0
mu = 0
beta = 100
U = 5.0
v = 1.0
D = 3
gamma = 0.0

# Parameters of the auxiliary system
Nb = 1
nsite = 2 * Nb + 1
Us = np.zeros(nsite)
Us[Nb] = U

#  Frequency grid
N_freq = 200
freq_max = 10
freq = np.linspace(-freq_max, freq_max, N_freq)

#  Selfconsistency conditions
error = 1e-4
mixing = 0.1
max_iter = 100

# ##################### Initializing Lindblad class ###########################
L = lind.Lindbladian(nsite,
                     Dissipator=lind.Dissipator_thermal_bath)
two_point_corr = corr.Correlators(L, 1)

# ##################### Initial Green's function of the system ################
args = np.array([e0, mu, beta, D, gamma], dtype=np.float64)
flat_hybridization = du.set_hybridization(
    freq, du.flat_bath_retarded, args)

args = np.array([e0, mu, beta, 1.0, 1.0], dtype=np.float64)
G_sys = du.set_hybridization(
    freq, du.lorenzian_bath_retarded, args)
G_tmp = du.set_hybridization(
    freq, du.lorenzian_bath_retarded, args)

optimization_options = {"disp": False, "maxiter": 500, 'ftol': 1e-5}

err = []
x_start = [0., 0.1, 0.5, -0.1, 0.2]
for i in range(max_iter):
    print(f"Iteration No.: {i}")

    # ######### Calculate the DMFT hybridization for a Bethe lattice. #########
    dmft_hyb = fg.FrequencyGreen(
        freq, G_sys.retarded * (v**2), G_sys.keldysh * (v**2))
    # ##################   Set up the total hybridization     #################
    hybridization = flat_hybridization + dmft_hyb

    # ## Optimization for determining the auxiliary hybridization function ####
    result_nb1 = opt.optimization_ph_symmertry(Nb, hybridization,
                                               x_start=x_start,
                                               options=optimization_options)
    x_start = np.copy(result_nb1.x)

    aux_sys = opt.get_aux(result_nb1.x, Nb, freq)
    hyb_aux = fg.get_hyb_from_aux(aux_sys)

    # ######## Calculate the auxiliary single particle Green's function #######
    T_mat = np.array(aux_sys.E)
    T_mat[Nb, Nb] = -U / 2.

    two_point_corr.update_model_parameter(
        np.array(aux_sys.Gamma1), np.array(aux_sys.Gamma2), T_mat, Us)
    two_point_corr.set_rho_steady_state()
    two_point_corr.sectors_exact_decomposition()

    G_plus, G_minus = two_point_corr.get_two_point_correlator_frequency(
        freq, two_point_corr.Lindbladian.liouville_ops.c(Nb, "up"),
        two_point_corr.Lindbladian.liouville_ops.cdag(Nb, "up"), (1, 0))

    G_R = G_plus + G_minus
    G_K = G_plus + np.conj(G_minus) - np.conj(G_plus + np.conj(G_minus))
    G_aux = fg.FrequencyGreen(aux_sys.ws, retarded=G_R, keldysh=G_K)

    # ##################### Extract the self-energy  ##########################
    sigma = G_aux.get_self_enerqy() - hyb_aux
    # ########### Calculate the system single particle Green's function #######
    G_sys.dyson(aux_sys.ws, hybridization + dmft_hyb + sigma)
    err.append(opt.cost_function(G_sys, G_tmp, normalize=False))
    if err[-1] < 1e-4:
        break
    print(f"Error is: {err[i]}")
    G_tmp = fg.FrequencyGreen(aux_sys.ws, retarded=((1.0 - mixing)
                                                    * G_sys.retarded
                                                    + mixing * G_tmp.retarded),
                              keldysh=((1.0 - mixing) * G_sys.keldysh
                              + mixing * G_tmp.keldysh))
    if i == 0:
        green = fg.FrequencyGreen(aux_sys.ws, G_aux.retarded, G_aux.keldysh)


plt.figure()
plt.plot(G_sys.freq, G_sys.retarded.imag)
plt.plot(G_aux.freq, G_aux.retarded.imag)
plt.xlabel(r"$\omega$")
plt.legend([r"$ImG^R_{sys}(\omega)$",
            r"$ImG^R_{aux}(\omega)$"
            ])

plt.figure()
plt.plot(G_sys.freq, G_sys.keldysh.imag)
plt.plot(G_sys.freq, hybridization.keldysh.imag)
plt.plot(G_sys.freq, sigma.keldysh.imag)
plt.xlabel(r"$\omega$")
plt.legend([r"$ImG^K_{sys}(\omega)$",
            r"$Im\Delta^K_{sys}(\omega)$",
            r"$Im\Sigma^K_{sys}(\omega)$"
            ])

plt.figure()
plt.plot(aux_sys.ws, green.retarded.imag)
plt.plot(aux_sys.ws, G_R.imag)
plt.xlabel(r"$\omega$")
plt.legend([r"$ImG^R_{aux,0}(\omega)$",
            r"$ImG^R_{aux}(\omega)$"])
plt.show()

plt.figure()
plt.plot(aux_sys.ws, green.keldysh.imag)
plt.plot(aux_sys.ws, G_K.imag)
plt.xlabel(r"$\omega$")
plt.legend([r"$ImG^K_{aux,0}(\omega)$",
            r"$ImG^K_{aux}(\omega)$"])
plt.show()

plt.figure()
plt.plot(aux_sys.ws, hyb_aux.retarded.imag)
plt.plot(aux_sys.ws, sigma.retarded.imag)
plt.xlabel(r"$\omega$")
plt.legend([r"$Im\Delta^R_{aux}(\omega)$",
            r"$Im\Sigma^R_{aux}(\omega)$"])
plt.show()
