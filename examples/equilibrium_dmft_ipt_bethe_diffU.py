# %%
import matplotlib.pyplot as plt
import numpy as np
import src.dmft.dmft_ipt as dmft

N_grid = 2001
freq_max = 15
time_max = 20
selfconsist_param = {'max_iter': 100, 'err_tol': 1e-6, 'mixing': 0.3}

e0 = 0
mu = 0
beta = 10
D = 30.1
gamma = 0.02

leads_param = {'e0': e0, 'mu': [mu], 'beta': beta, 'D': D, 'gamma': gamma}

errors = []
Us = [0, 3, 5, 7, 9]
for U in Us:
    print("U: ", U)
    v = 1.0
    sys_param = {"e0": 0, 'v': v, 'U': U}

    # Parameters of the auxiliary system
    Nb = 1
    nsite = 2 * Nb + 1
    aux_param = {'Nb': Nb, 'nsite': nsite}

    params = {'freq': {"freq_min": -freq_max, "freq_max": freq_max,
                       'N_freq': N_grid},
              'time': {"time_min": -time_max, "time_max": time_max,
                       "N_time": N_grid},
              'selfconsistency': selfconsist_param, 'leads': leads_param,
              'aux_sys': aux_param, 'system': sys_param}

    # ################### Initializing Lindblad class ####################

    dmft_ipt = dmft.DMFT_IPT(params, hyb_leads=None)
    dmft_ipt.hyb_leads = dmft_ipt.get_bath()
    dmft_ipt.solve()

    i_min = np.where(dmft_ipt.green_sys.freq > -10)[0][0]
    i_max = np.where(dmft_ipt.green_sys.freq > 10)[0][0]
    plt.plot(dmft_ipt.green_sys.freq[i_min: i_max], -(1 / np.pi)
             * dmft_ipt.green_sys.retarded.imag[i_min: i_max], label=U)
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$A(\omega)$")
    plt.legend()
    errors.append(dmft_ipt.err_iterations)
plt.show()
for i, U in enumerate(Us):
    plt.plot(errors[i], label=U)
    plt.yscale("log")
    plt.xlabel(r"Iteration")
    plt.ylabel(r"$||G_{new}(\omega)-G_{old}(\omega)||$")
    plt.legend()
plt.show()

# %%
