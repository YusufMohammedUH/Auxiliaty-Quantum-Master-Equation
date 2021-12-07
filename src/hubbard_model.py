#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:15:29 2021

@author: Matteo Vandelli
"""
import numpy as np
from itertools import product
from src.define_states_fock import basis_state, dot_prod, op_chain, cd_js, c_js


def Hubbard_atom(U, mu):
    """
    Generation of the Hubbard Hamiltonian for a set of sites.
    Hoppings and chemical potential assumed to be spin-independent.

    Parameters
    ----------
    U : float
        Hubbard interaction.
    mu : float
        Chemical potential.
    Returns
    -------
    Hamil : np.ndarray (4, 4)
            Hamiltonian in Fock space.
    states : list of state objects
             contains the set of basis states.
    """

    # It has 4 states
    empty_state = basis_state(1, 2, np.array([[0, 0]]))
    spin_up = basis_state(1, 2, np.array([[0, 1]]))
    spin_do = basis_state(1, 2, np.array([[1, 0]]))
    spin_updo = basis_state(1, 2, np.array([[1, 1]]))

    # These are also eigenstates of the Hamiltonian with
    # eigenvalues: 0, -mu, -mu, U - 2*mu

    states = [empty_state, spin_up, spin_do, spin_updo]
    print("Hilbert space is ", states)
    n_states = len(states)
    Hamil = np.zeros((n_states, n_states))

    for n1 in np.arange(n_states):
        for n2 in np.arange(len(states)):
            nup_state = op_chain(states[n1], ((cd_js, 0, 1), (c_js, 0, 1)))
            ndo_state = op_chain(states[n1], ((cd_js, 0, 0), (c_js, 0, 0)))
            nn_state = op_chain(states[n1], ((cd_js, 0, 1), (c_js, 0, 1),
                                             (cd_js, 0, 0), (c_js, 0, 0)))
            Hamil[n1, n2] = U * dot_prod(states[n2], nn_state) - mu * (
                dot_prod(states[n2], nup_state)
                + dot_prod(states[n2], ndo_state))
    return Hamil, states


def Hubbard_chain(t, U, mu, n_sites):
    """
    Generation of the Hubbard Hamiltonian for a set of sites.
    Hoppings and chemical potential assumed to be spin-independent.

    Parameters
    ----------
    t : numpy.ndarray (n_sites, n_sites)
        Hopping between different sites.
    U : numpy.ndarray (n_sites,)
        Hubbard interaction on each site.
    mu : numpy.ndarray (n_sites,)
        Chemical potential on each site.
    n_sites : int
        number of sites.

    Returns
    -------
    Hamil : np.ndarray (4**n_sites, 4**n_sites)
            Hamiltonian in Fock space.
    states : list of state objects
             contains the set of basis states.
    """
    # define Hubbard atom
    n_spins = 2

    # It has 4 states
    states = []
    for conf in product([0, 1], repeat=n_spins * n_sites):
        conf = np.asarray(conf)
        occup = conf.reshape(n_sites, n_spins)
        state_in = basis_state(n_sites, n_spins, occup)
        # excl = False
        # for site in np.arange(2):
        #     if occup[site, 0] == 1 and occup[site, 1] == 1:
        #         excl = True
        #     if (occup == 0).all():
        #         excl = True
        # if excl:
        #     continue
        states.append(state_in)

    n_states = len(states)
    Hamil = np.zeros((n_states, n_states))

    for n1 in np.arange(n_states):
        for n2 in np.arange(len(states)):
            # Local contribution
            for site in np.arange(n_sites):
                nup_state = op_chain(states[n1], ((cd_js, site, 1),
                                                  (c_js, site, 1)))
                ndo_state = op_chain(states[n1], ((cd_js, site, 0),
                                                  (c_js, site, 0)))
                nn_state = op_chain(states[n1], ((cd_js, site, 1),
                                                 (c_js, site, 1),
                                                 (cd_js, site, 0),
                                                 (c_js, site, 0)))
                Hamil[n1, n2] += U[site] * dot_prod(states[n2], nn_state)
                Hamil[n1, n2] -= mu[site] * (dot_prod(states[n2], nup_state)
                                             + dot_prod(states[n2], ndo_state))
            # Non-local contribution
            for site1 in np.arange(n_sites):
                for site2 in np.arange(n_sites):
                    for sp in np.arange(n_spins):
                        state_next = op_chain(states[n1], ((cd_js, site1, sp),
                                                           (c_js, site2, sp)))
                        Hamil[n1, n2] += t[site1, site2] * dot_prod(states[n2],
                                                                    state_next)
    return Hamil, states
