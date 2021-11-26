#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:46:16 2021

@author: Matteo Vandelli

Application of the defined routines to the 2-level system.
"""
import numpy as np
from define_states_liouville import Liouville_basis_state
from define_states_liouville import c_js_F, c_js_T, cd_js_F, cd_js_T
from define_states_liouville import dot_prod_LF
from define_states_fock import kron_delta


def create_Liouvillian_2level(W, epsilon0, epsilon1, Gamma):
    # The two-level system is equivalent to spin
    # so fix lattice sites to 1 and orbital sites to 2
    n_sites = 1
    n_orbs = 2

    # create the two-level system problem
    Gamma_m = Gamma * np.array([[0., 0.], [0., 1.]])
    Hamil = np.array([[epsilon0, W], [W, epsilon1]])

    # List of Fock-Liouville states
    FL_states = []

    # create single-particle states for spinful site
    # equivalent to two-level system
    for site in np.arange(n_sites):
        for sp in np.arange(n_orbs):
            occup = np.zeros((n_sites, n_orbs), dtype=int)
            occup[site, sp] = 1
            for site_t in np.arange(n_sites):
                for sp_t in np.arange(n_orbs):
                    occup_t = np.zeros((n_sites, n_orbs), dtype=int)
                    occup_t[site_t, sp_t] = 1
                    FL_state = Liouville_basis_state(
                        n_sites, n_orbs, occup, occup_t)
                    # print(site, sp, site_t, sp_t, FL_state)
                    FL_states.append(FL_state)

    # Create Liouvillian operator
    Liouvillian = np.zeros((n_orbs**2, n_orbs**2), complex)
    # n1 and n2 are Fock space indices
    for n1 in np.arange(n_orbs):
        for n2 in np.arange(n_orbs):
            # n3 and n4 are tilde space indices
            for n3 in np.arange(n_orbs):
                for n4 in np.arange(n_orbs):
                    # outer states for bra-ket matrix elements
                    state_in = FL_states[n_orbs * n1 + n3]
                    state_out = FL_states[n_orbs * n2 + n4]
                    # Construction of the Hamiltonian part
                    cdc_F = cd_js_F(0, n2, c_js_F(0, n1, state_in))
                    cdc_T = cd_js_T(0, n4, c_js_T(0, n3, state_in))
                    dp_F = dot_prod_LF(state_out, cdc_F)
                    dp_T = dot_prod_LF(state_out, cdc_T)
                    Liouvillian[n_orbs * n2 + n4, n_orbs * n1 + n3] = -1j * (
                        Hamil[n1, n2]*dp_F - Hamil[n3, n4]*dp_T)
                    # elements of the dissipator
                    # Sigma^+ {Sigma^-} and \tilde{Sigma^+} \tilde{Sigma^-}
                    Liouvillian[n_orbs * n2 + n4, n_orbs * n1 + n3] += \
                        - 0.5 * (Gamma_m[n1, n2] + Gamma_m[n3, n4]) * dot_prod_LF(state_out, state_in)
                    # Sigma^- \tilde{Sigma^-}
                    Liouvillian[n_orbs * n2 + n4, n_orbs * n1 + n3] += \
                        Gamma * kron_delta(n1, n2+1) * kron_delta(n3, n4+1)
    return Liouvillian


if __name__ == "__main__":
    W = 1.
    epsilon0 = 0.
    epsilon1 = 0.3
    Gamma = 0.2
    Liouvillian = create_Liouvillian_2level(W, epsilon0, epsilon1, Gamma)
    print("L = ", Liouvillian)

    # TEST: compare with the analytical expression
    # Analytical expression  for the Liouvillian
    # Liouvillian for 2-level system
    Liouville = np.array([[0., 1j*W, -1j*W, Gamma],
                          [1j*W, 1j*epsilon1 - 0.5*Gamma, 0., -1j*W],
                          [-1j*W, 0, -1j*epsilon1 - 0.5*Gamma, 1j*W],
                          [0., -1j*W, 1j*W, -Gamma]
                          ])
    assert np.all(Liouvillian == Liouville)
