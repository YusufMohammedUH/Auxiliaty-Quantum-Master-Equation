#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 13:34:38 2021

@author: Matteo Vandelli


Definition of the states, creation and annihilation operators in
Fock space for small systems.
No optimization is performed here, as it is usually NOT a performance critical
section.
"""
import numpy as np
from copy import deepcopy
# from numba import jit --> CI does not like numba


# @jit(nopython=True)
def kron_delta(a, b):
    """
    Given two objects a and b with a == operator defined,
    return 1 if they are equal, 0 otherwise.
    """
    if a == b:
        return 1
    else:
        return 0

###########################################################


class state:
    """
    Class for creation of a state in Fock space.
    Inputs :
    -------------
        n_site : int, number of sites
        n_orb : int, number of local orbitals/spin
        occup : numpy.ndarray (len, orbs) with elements
                    -1 (null state), 0 (empty) or 1 (full)
    Members :
    -------------
        len : int, number of sites
        orbs : int, number of local orbitals/spin
        occup : numpy.ndarray (len, orbs) with elements
                -1 (null state), 0 (empty) or 1 (full)
    """

    def __init__(self, n_sites, n_orbs, occup):
        self.len = n_sites
        self.orbs = n_orbs
        self.occup = deepcopy(occup)
        for n in np.arange(self.len):
            for sp in np.arange(self.orbs):
                occ = self.occup[n, sp]
                if occ != 0 and occ != 1 and occ != -1:
                    raise ValueError("fermionic n has to be either 0 or 1")

    def __str__(self):
        """
        Overloading string operator to write state in occupation basis
        |n_{i, s1} ... n_{i, sn}>.
        Return a string.
        """
        out_str = []
        for n in np.arange(self.len):
            out_str.append("|")
            for sp in np.arange(self.orbs):
                out_str.append(str(self.occup[n, sp]))
            out_str.append(">")
        out_str = "".join(out_str)
        return out_str

    def __repr__(self):
        """
        Overloading repr operator to write state in occupation basis
        |n_{i, s1} ... n_{i, sn}>.
        Return a string.
        """
        out_str = []
        for n in np.arange(self.len):
            out_str.append("|")
            for sp in np.arange(self.orbs):
                out_str.append(str(self.occup[n, sp]))
            out_str.append(">")
        out_str = "".join(out_str)
        return out_str

#########################################################


def dot_prod(bra: state, ket: state):
    """
    Given two Fock states of type state, return their dot product.
    """
    if (not isinstance(bra, state)) or (not isinstance(ket, state)):
        raise TypeError(
            "states must be of type {}".format(
                state))
    if np.any(ket.occup == -1) or np.any(bra.occup == -1):
        return 0.
    assert ket.orbs == bra.orbs and ket.len == bra.len
    result = 1.
    for n in np.arange(ket.len):
        for sp in np.arange(ket.orbs):
            result *= kron_delta(bra.occup[n, sp], ket.occup[n, sp])
    return result

##########################################################

# I allow operators only to be applied to the right


def c_js(j, sigma, ket_in):
    r"""
    Given a site index j, spin sigma and a state ket_in,
    return the state c_{js}|in> = |n_0, ... (n-1)_{j, sigma}, ...>.
    """
    ket_out = deepcopy(ket_in)
    if not isinstance(ket_out, state):
        raise TypeError(
            "ket_out must be of type {} respectively".format(
                state))
    if ket_out.occup[j, sigma] == -1:
        return ket_out

    elif ket_out.occup[j, sigma] == 0 or ket_out.occup[j, sigma] == 1:
        ket_out.occup[j, sigma] -= 1
    else:
        raise ValueError(
            "occup of states must be -1 (null), 0 (empty) or 1 (full)")
    return ket_out


def cd_js(j, sigma, ket_in):
    r"""
    Given a site index j, spin sigma and a state ket_in,
    return the state c^{\dagger}_{js}|in> = |n_0, ... (n+1)_{j, sigma}, ...>.
    """
    ket_out = deepcopy(ket_in)
    if not isinstance(ket_out, state):
        raise TypeError(
            "ket_out must be of type {} respectively".format(
                state))
    if ket_out.occup[j, sigma] == -1:
        return ket_out
    elif ket_out.occup[j, sigma] == 0 or ket_out.occup[j, sigma] == 1:
        ket_out.occup[j, sigma] = 1 - 2 * ket_out.occup[j, sigma]
    else:
        raise ValueError(
            "occup of states must be -1 (null), 0 (empty) or 1 (full)")
    return ket_out


def op_chain(state_in, oper_list):
    """
    Apply several operators one after the other to state_in.
    Each element of oper_list is a list with:
        - function : cd_js or c_js
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

###########################################################


if __name__ == "__main__":
    # test on a chain of atoms without periodic boundary conditions
    n_sites = 6
    n_orbs = 1
    states_in = []
    for site in np.arange(n_sites):
        for sp in np.arange(n_orbs):
            occup = np.zeros((n_sites, n_orbs), dtype=int)
            occup[site, sp] = 1
            states_in.append(state(n_sites, n_orbs, occup))
            print(site, sp, states_in[-1])
    """
    states_out = deepcopy(states_in)
    for st1 in states_out:
        for st2 in states_in:
            print(st1, st2, dot_prod(st1, st2))
    """
    matrix = np.zeros((n_sites, n_sites))
    for n1 in np.arange(n_sites):
        for n2 in np.arange(n_sites):
            if abs(n1 - n2) == 1:
                occ1 = np.zeros((n_sites, n_orbs), dtype=int)
                occ2 = np.zeros((n_sites, n_orbs), dtype=int)
                occ1[n1, 0] = 1
                occ2[n2, 0] = 1
                state1 = state(n_sites, n_orbs, occ1)
                state2 = state(n_sites, n_orbs, occ2)
                # state1 = cd_js(n2, 0, c_js(n1, 0, state1))
                state1 = op_chain(state1, [[cd_js, n2, 0], [c_js, n1, 0]])
                matrix[n1, n2] = - dot_prod(state1, state2)
    print("Hopping matrix =", matrix)
