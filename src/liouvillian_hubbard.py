#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:15:29 2021

@author: Matteo Vandelli
"""
import numpy as np
import matplotlib.pyplot as plt
from define_states_fock import kron_delta
from state_vector import create_Hubbard_basis_LF
from define_states_liouville import dot_prod_LF
from define_states_liouville import cd_js_F, c_js_F, cd_js_T, c_js_T, op_chain_LF
from liouville_decomposition import decompose_Liouville


def Hubbard_hamil_LF(space, states, t, U, mu, n_sites):
    """
    Hamiltonian in Lioville-Fock space. It can either be the tilde Hamiltonian
    or the Fock Hamiltonian. They will be equivalent, but for clarity
    we introduce the tilde parameter.

    Parameters
    ----------
    space : string, "tilde" or "fock"
        wheter the Hamiltonian is is tilde or in fock space.
    states : list of Liouville_state objects,
        basis states for the Hamiltonian.
    t : numpy.ndarray (n_sites, n_sites)
        Hopping between different sites.
    U : numpy.ndarray (n_sites,)
        Hubbard interaction on each site.
    mu : numpy.ndarray (n_sites,)
        Chemical potential on each site.
    n_sites : int
        number of sites.
    n_spins : int, default=2
        number of spin-orbitals.

    Returns
    -------
    Hamil : np.ndarray (4**(2*n_sites), 4**(2*n_sites))
            Hamiltonian in Liouville-Fock space.
    states : list of Liouville_state objects
             contains the set of basis states.

    """
    n2_states = len(states)
    Hamil = np.zeros((n2_states, n2_states), complex)
    if space == "tilde":
        c_dag = cd_js_T
        c_ann = c_js_T
        # tilde = 1
    else:
        c_dag = cd_js_F
        c_ann = c_js_F
        # tilde = 0
    for n1 in np.arange(n2_states):
        for n2 in np.arange(n2_states):
            # Local contribution
            for site in np.arange(n_sites):
                nup_state = op_chain_LF(states[n1], ((c_dag, site, 1),
                                                     (c_ann, site, 1)))
                ndo_state = op_chain_LF(states[n1], ((c_dag, site, 0),
                                                     (c_ann, site, 0)))
                nn_state = op_chain_LF(states[n1], ((c_dag, site, 1),
                                                    (c_ann, site, 1),
                                                    (c_dag, site, 0),
                                                    (c_ann, site, 0)))
                Hamil[n1, n2] += U[site] * dot_prod_LF(states[n2], nn_state)
                Hamil[n1, n2] += (- mu[site]) * (
                  dot_prod_LF(states[n2], nup_state)
                  + dot_prod_LF(states[n2], ndo_state)
                  )

            # Non-local contribution
            for site1 in np.arange(n_sites):
                for site2 in np.arange(n_sites):
                    for sp in [0, 1]:
                        state_next = op_chain_LF(states[n1],
                                                 ((c_dag, site1, sp),
                                                  (c_ann, site2, sp)))
                        Hamil[n1, n2] += t[site1, site2] * dot_prod_LF(
                            states[n2],
                            state_next)
    return Hamil


def Dissip_Naka_LF(states, Gamma, n_sites, n_spins=2):
    """
    Dissipator obtained from Nakagawa et al. arXiv: 2003.14202
    Problem: this form of the dissipator only destroys two-particles.
    This means that there are several steady states, i.e. all the states
    made by |00>,|01> and |10>. In fact there are 9 of them.
    Parameters
    ----------
    states : TYPE
        DESCRIPTION.
    Gamma : TYPE
        DESCRIPTION.
    n_sites : TYPE
        DESCRIPTION.
    n_spins : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    Dissip : TYPE
        DESCRIPTION.

    """
    n2_states = len(states)
    Dissip = np.zeros((n2_states, n2_states), complex)
    for n1 in np.arange(n2_states):
        for n2 in np.arange(n2_states):
            # Local contribution
            for site in np.arange(n_sites):
                ff_state = op_chain_LF(states[n1], ((cd_js_F, site, 1),
                                                    (c_js_F, site, 1),
                                                    (cd_js_F, site, 0),
                                                    (c_js_F, site, 0)))
                tt_state = op_chain_LF(states[n1], ((cd_js_T, site, 1),
                                                    (c_js_T, site, 1),
                                                    (cd_js_T, site, 0),
                                                    (c_js_T, site, 0)))
                # nup_state = op_chain_LF(states[n1], ((cd_js_T, site, 1),
                #                                      (c_js_T, site, 1)))
                # ndo_state = op_chain_LF(states[n1], ((cd_js_T, site, 0),
                #                                      (c_js_T, site, 0)))
                Dissip[n1, n2] += Gamma[site] * (
                    - 1j*dot_prod_LF(states[n2], ff_state)
                    - 1j*dot_prod_LF(states[n2], tt_state))
                # Dissip[n1, n2] += - 1j*Gamma[site] * (
                #    dot_prod_LF(states[n2], nup_state)
                #    + dot_prod_LF(states[n2], ndo_state)) \
                #    -  1j * Gamma[site] * kron_delta(n1, n2)
    return -1j*Dissip


def Dissip_Dorda_LF(states, Gamma, n_sites, n_spins=2):
    n2_states = len(states)
    Gamma1, Gamma2 = Gamma
    Dissip = np.zeros((n2_states, n2_states), complex)
    for n1 in np.arange(n2_states):
        for n2 in np.arange(n2_states):
            # Local contribution
            for site1 in np.arange(n_sites):
                for site2 in np.arange(n_sites):
                    for sp2 in np.arange(n_spins):
                            # cd_T c_T
                            tt_state = op_chain_LF(states[n1],
                                                    ((cd_js_T, site2, sp2),
                                                    (c_js_T, site2, sp2)))
                            # cd_T cd_F
                            tf_state = op_chain_LF(states[n1],
                                                    ((c_js_T, site2, sp2),
                                                    (c_js_F, site2, sp2)))
                            # c_T c_F
                            ft_state = op_chain_LF(states[n1],
                                                    ((cd_js_F, site2, sp2),
                                                    (cd_js_T, site2, sp2)))

                            # cd_T cd_F
                            ff_state = op_chain_LF(states[n1],
                                                    ((cd_js_F, site2, sp2),
                                                    (c_js_F, site2, sp2)))
                            G1_m_G2 = Gamma1[site1, site2] - Gamma2[site1, site2]
                            Dissip[n1, n2] -= 1j * G1_m_G2 * dot_prod_LF(
                                                states[n2], ff_state)
                            Dissip[n1, n2] -= 1j*G1_m_G2 * dot_prod_LF(
                                                states[n2], tt_state)
                            Dissip[n1, n2] += 2*Gamma2[site1, site2] * \
                                dot_prod_LF(states[n2], ft_state)
                            Dissip[n1, n2] -= 2*Gamma1[site1, site2] * \
                                dot_prod_LF(states[n2], tf_state)
                            Dissip[n1, n2] -= 2.j * np.trace(Gamma2) * kron_delta(n1, n2)
    return -1j*Dissip


def naive_Liouvillian_Hubbard(t, Gamma, U, mu,
                              n_sites, n_spins=2, Dissip=None):
    """
    Generation of the Hubbard Hamiltonian for a set of sites.
    Hoppings and chemical potential assumed to be spin-independent.

    Parameters
    ----------
    t : numpy.ndarray (n_sites, n_sites)
        Hopping between different sites.
    Gamma1, Gamma2 : numpy.ndarray (n_sites, n_sites)
        Lindblad coupling matrices of empyty/full baths.
    U : numpy.ndarray (n_sites,)
        Hubbard interaction on each site.
    mu : numpy.ndarray (n_sites,)
        Chemical potential on each site.
    n_sites : int
        number of sites.
    Dissip : funct or None, default is None
        if funct, compute the dissipator using that function.
        if None, no dissipator included here.

    Returns
    -------
    Hamil : np.ndarray (4**n_sites, 4**n_sites)
            Hamiltonian in Fock space.
    states : list of state objects
             contains the set of basis states.
    """
    # define Hubbard atom

    # It has 4 states
    states = create_Hubbard_basis_LF(n_sites, n_spins)
    Hamil_F = Hubbard_hamil_LF("fock", states, t, U, mu, n_sites)
    Hamil_T = Hubbard_hamil_LF("tilde", states, t, U, mu, n_sites)
    Liouville = -1j*(Hamil_F - Hamil_T)
    # TODO add here the dissipator
    if not(Dissip is None):
        Liouville += Dissip(states, Gamma, n_sites, n_spins)
    return Liouville, states


def find_nearest(a, a0):
    """Element in nd array `a` closest to the scalar value `a0`,
    returns the index"""
    idx = np.abs(a - a0).argmin()
    return idx


if __name__ == "__main__":
    t = 0.0 * np.array([[1.]])
    mu = - 1.0 * np.array([[1.]])
    U = 0.0 * np.array([[1.]])
    Gamma1 = 0.2 * np.array([[1.]])
    Gamma2 = 0.3 * np.array([[1.]])
    Liouville_sl, states = naive_Liouvillian_Hubbard(
        t, (Gamma1, Gamma2), U, mu, 1, n_spins=2, Dissip=Dissip_Dorda_LF)
    print("States : ", states)
    vals, vecs_l, vecs_r = decompose_Liouville(Liouville_sl)
    print("Eigenvalues :", np.sort(vals))
    plt.figure()
    plt.title("Eigenvalues of the Liouvillian")
    for val in vals:
        plt.scatter(vals.real, vals.imag)
    plt.xlabel(r"$Re \, \Lambda$")
    plt.ylabel(r"$Im \, \Lambda$")
    idx = find_nearest(vals, 0.0)
    assert np.allclose(vals[idx], 0.0)
    print(vals[idx])
    plt.show()
    
    
