"""Example for auxliliary quantum master equation dual gw with DMFT
    self-consistency.
"""
# %%
import matplotlib.pyplot as plt
import src.super_fermionic_space.super_fermionic_subspace as sf_sub
import src.super_fermionic_space.model_lindbladian as lind
import src.greens_function.correlation_functions as corr
import src.dmft.aqm_dmft as aux_dmft
import src.dmft.aqm_dual_gw_dmft as aqm_gw_dmft_dmft
# ############################### Parameters ##################################
#  Frequency grid
N_freq = 401
freq_max = 10

N_time = 1000
time_max = 20

selfconsist_param = {'max_iter': 50, 'err_tol': 1e-7, 'mixing': 0.2}

e0 = 0
mu = 0
beta = 1000
D = 10
gamma = 0.01

leads_param = {'e0': e0, 'mu': [mu], 'beta': beta, 'D': D, 'gamma': gamma}

spinless = False
spin_sector_max = 2
tilde_conjugationrule_phase = True

v = 1.0
sys_param = {'e0': 0, 'v': v, 'orbitals': 1, 'spinless': spinless,
             'tilde_conjugation': tilde_conjugationrule_phase, 'keldysh_comp':
             'keldysh'}

# Parameters of the auxiliary system
Nb = 1
nsite = 2 * Nb + 1
aux_param = {'Nb': Nb, 'nsite': nsite}

dual_GW = {'err_tol': 1e-7, 'iter_max': 50}

params = {'time': {"time_min": -time_max, "time_max": time_max,
                   'N_time': N_time},
          'freq': {"freq_min": -freq_max, "freq_max": freq_max,
                   'N_freq': N_freq},
          'dual_GW': dual_GW,
          'selfconsistency': selfconsist_param, 'leads': leads_param,
          'aux_sys': aux_param, 'system': sys_param}
super_fermi_ops = sf_sub.SpinSectorDecomposition(
    nsite, spin_sector_max, spinless=spinless,
    tilde_conjugationrule_phase=tilde_conjugationrule_phase)

L = lind.Lindbladian(super_fermi_ops=super_fermi_ops)
corr_cls = corr.Correlators(L, trilex=True)

# ########################## Solve Auxiliary DMFT #############################
plt.figure()
err_U = {}
for U in [0., 1., 2., 3., 4.]:
    print(f"Runnig auxiliary DMFT with U = {U}")
    params['system']['U'] = U
    aqm_gw_dmft = aqm_gw_dmft_dmft.AuxiliaryMaserEquationDualGWDMFT(
        params, corr_cls)
    aqm_gw_dmft.hyb_leads = aqm_gw_dmft.get_bath()
    aqm_gw_dmft.set_local_matrix()
    aqm_gw_dmft.solve()

    aqm_dmft = aux_dmft.AuxiliaryMaserEquationDMFT(params, corr_cls)
    aqm_dmft.hyb_leads = aqm_dmft.get_bath()
    aqm_dmft.set_local_matrix()
    aqm_dmft.solve()
    err_U[U] = aqm_gw_dmft.err_iterations
    plt.plot(aqm_gw_dmft.green_sys.freq,
             aqm_gw_dmft.green_sys.get_spectral_func())
    plt.plot(aqm_dmft.green_sys.freq,
             aqm_dmft.green_sys.get_spectral_func())
    plt.show()
# plt.legend([r"$U = 0$", r"$U = 1$", r"$U = 2$", r"$U = 3$", r"$U = 4$"])
# plt.ylabel(r"$A(\omega)$")
# plt.xlabel(r"$\omega$")
# plt.show()
# ################################## Plots ####################################
# %%
plt.figure()
for U in [0., 1., 2., 3., 4.]:
    plt.plot(err_U[U])
plt.legend([r"$U = 0$", r"$U = 1$", r"$U = 2$", r"$U = 3$", r"$U = 4$"])
plt.yscale('log')
plt.show()

plt.plot(aqm_gw_dmft.green_sys.freq,
         aqm_gw_dmft.green_sys.get_spectral_func())
plt.plot(aqm_gw_dmft.green_sys.freq,
         aqm_gw_dmft.green_aux.get_spectral_func())
plt.xlabel(r"$\omega$")
plt.legend([r"$A_{sys}(\omega)$",
            r"$A_{aux}(\omega)$"
            ])

plt.figure()
plt.plot(aqm_gw_dmft.green_sys.freq, aqm_gw_dmft.green_sys.keldysh.imag)
plt.plot(aqm_gw_dmft.green_sys.freq, aqm_gw_dmft.hyb_leads.keldysh.imag)
plt.plot(aqm_gw_dmft.green_sys.freq,
         aqm_gw_dmft.self_energy_int.keldysh.imag)
plt.xlabel(r"$\omega$")
plt.legend([r"$ImG^K_{sys}(\omega)$",
            r"$Im\Delta^K_{sys}(\omega)$",
            r"$Im\Sigma^K_{sys}(\omega)$"
            ])

plt.figure()
plt.plot(aqm_gw_dmft.green_sys.freq, aqm_gw_dmft.hyb_aux.retarded.imag)
plt.plot(aqm_gw_dmft.green_sys.freq,
         aqm_gw_dmft.self_energy_int.retarded.imag)
plt.xlabel(r"$\omega$")
plt.legend([r"$Im\Delta^R_{aux}(\omega)$",
            r"$Im\Sigma^R_{aux}(\omega)$"])
plt.show()

# %%
