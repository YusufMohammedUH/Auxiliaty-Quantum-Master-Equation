#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 17:05:44 2021

@author: Matteo Vandelli

Definition of the states and operators in Liouville-Fock space, taking
advantage of the previously defined subroutines.
"""
import numpy as np
from copy import deepcopy
from define_states_fock import state, c_js, cd_js, dot_prod


class Liouville_state:
    def __init__(self, n_site, n_orb, occup, occup_tilde):
        """
        Class for creation of a state in Fock space.
        Inputs :
        -------------
            n_site : int, number of sites
            n_orb : int, number of local orbitals/spin
            occup : numpy.ndarray (len, orbs) with elements
                    -1 (null state), 0 (empty) or 1 (full)
            occup_t : numpy.ndarray (len, orbs) with elements
                    -1 (null state), 0 (empty) or 1 (full)
        Members :
        --------------
             st_fock : state object, Fock component of the state
             st_tilde : state object, Tilde component of the state
        """

        self.st_fock = state(n_site, n_orb, occup)
        self.st_tilde = state(n_site, n_orb, occup_tilde)

    def __repr__(self):
        return repr(self.st_fock) + " x " + repr(self.st_tilde)


def oper_js_FL(oper, j, sigma, tilde, FL_state):
    """
    Apply an operator oper acting on the Fock-Liouville state FL_state
    site j, spin sigma and in the space defined
    by tilde (Fock/Tilde --> True/False).
    Return a new FL_state.
    """
    ket_in = deepcopy(FL_state)
    if tilde:
        ket_in.st_tilde = oper(j, sigma, ket_in.st_tilde)
    else:
        ket_in.st_fock = oper(j, sigma, ket_in.st_fock)
    return ket_in


def c_js_F(j, sigma, FL_state):
    """
    Apply the annihilation operator on the Fock-Liouville state FL_state
    site j, spin sigma in the original Fock space.
    Return a new FL_state.
    """
    return oper_js_FL(c_js, j, sigma, False, FL_state)


def c_js_T(j, sigma, FL_state):
    """
    Apply the annihilation operator on the Fock-Liouville state FL_state
    site j, spin sigma in the tilde space.
    Return a new FL_state.
    """
    return oper_js_FL(c_js, j, sigma, True, FL_state)


def cd_js_F(j, sigma, FL_state):
    """
    Apply the creation operator on the Fock-Liouville state FL_state
    site j, spin sigma in the original Fock space.
    Return a new FL_state.
    """
    return oper_js_FL(cd_js, j, sigma, False, FL_state)


def cd_js_T(j, sigma, FL_state):
    """
    Apply the creation operator on the Fock-Liouville state FL_state
    site j, spin sigma in the tilde space.
    Return a new FL_state.
    """
    return oper_js_FL(cd_js, j, sigma, True, FL_state)


def op_chain_LF(state_in, oper_list):
    """
    Apply several operators one after the other to state_in.
    Each element of oper_list is a list with:
        - function : cd_js_T, cd_js_F, c_js_T or c_js_F
        - site : int, site index
        - orb : int, spin index
    They are applied in the reversed order (to the right).
    Return a state object.
    """
    state1 = deepcopy(state_in)
    for arg in oper_list[::-1]:
        oper, site, spin = arg
        state1 = oper(site, spin, state1)
    return state1


def dot_prod_LF(state_out, state_in):
    """
    Given two LF_state objects, compute the dot product between them
    and return the corresponding value.
    """
    dotp_F = dot_prod(state_out.st_fock, state_in.st_fock)
    dotp_T = dot_prod(state_out.st_tilde, state_in.st_tilde)
    return dotp_F * dotp_T


if __name__ == "__main__":
    # Some simple testing:
    # prints the single-electron states on two sites in Liouville-Fock space.
    # this prints: |n_1 n_2> x |n_1 n_2> (first is Fock, second is Tilde):
    # |10> x |10>
    # |10> x |01>
    # |01> x |10>
    # |01> x |01>
    # Additionally it prints the result from application
    # of some c/c^{\dagger} operators.

    n_sites = 2
    n_orbs = 1
    states_in = []
    j = 0
    sigma = 0
    for site in np.arange(n_sites):
        for sp in np.arange(n_orbs):
            occup = np.zeros((n_sites, n_orbs), dtype=int)
            occup[site, sp] = 1
            for site_t in np.arange(n_sites):
                for sp_t in np.arange(n_orbs):
                    occup_t = np.zeros((n_sites, n_orbs), dtype=int)
                    occup_t[site_t, sp_t] = 1
                    FL_state = Liouville_state(n_sites, n_orbs, occup, occup_t)
                    print("State", FL_state)
                    print(r"c_{00}"+repr(FL_state)+"=",
                          c_js_F(j, sigma, FL_state))
                    print(r"cd_{00}"+repr(FL_state)+"=",
                          cd_js_F(j, sigma, FL_state))
                    print(r"\tilde{c}_{00}"+repr(FL_state)+"=",
                          c_js_T(j, sigma, FL_state))
                    print(r"\tilde{cd}_{00}"+repr(FL_state)+"=",
                          cd_js_T(j, sigma, FL_state))
                    print("")
