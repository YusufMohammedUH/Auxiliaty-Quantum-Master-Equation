#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:06:40 2021

@author: Matteo Vandelli

Here I construct the actual vector in Hilbert space.
"""

import numpy as np
from copy import deepcopy
from itertools import product
from src.define_states_liouville import op_chain_LF, \
    dot_prod_LF, Liouville_basis_state, c_js_F, c_js_T, cd_js_F, cd_js_T
from src.define_states_fock import op_chain, dot_prod, basis_state


class LF_vector:
    def __init__(self, LF_basis, weights):
        """
        Unnormalized vector in the Hilbert state.
        Contains the basis states and the weight associated to it.

        Parameters
        ----------
        LF_states : list of (Liouville_)basis_state, len = n_states
            states of the Hilbert space.
        weights : numpy.array (n_states,)
            weight of each basis vector to the linear combination.

        Raises
        ------
        ValueError
            if states and weights do not have the same length.
        """
        if len(LF_basis) != weights.shape[0]:
            raise ValueError("States and weights must have same length.")
        self.n_states = len(LF_basis)
        self.basis = LF_basis
        self.weights = weights

    def __add__(self, LF_vector_2):
        assert repr(self.basis) == repr(LF_vector_2.basis)
        weights_p = LF_vector_2.weights + self.weights
        basis = deepcopy(self.basis)
        return LF_vector(basis, weights_p)

    def __sub__(self, LF_vector_2):
        assert repr(self.basis) == repr(LF_vector_2.basis)
        weights_p = self.weights - LF_vector_2.weights
        basis = deepcopy(self.basis)
        return LF_vector(basis, weights_p)

    def __mul__(self, rhs):
        if isinstance(rhs, (int, float, complex)):
            weights = rhs * deepcopy(self.weights)
            return LF_vector(self.basis, weights)
        assert repr(self.basis) == repr(rhs.basis)
        rhs_ = deepcopy(rhs)
        weight = np.sum(rhs_.weights * np.conj(self.weights))
        return weight

    def __rmul__(self, lhs):
        if isinstance(lhs, (int, float, complex)):
            weights = lhs * deepcopy(self.weights)
            return LF_vector(self.basis, weights)
        return np.conj(self.__mul__(lhs))

    def __neg__(self):
        LF_vec_neg = LF_vector(deepcopy(self.basis), - deepcopy(self.weights))
        return LF_vec_neg

    def __repr__(self):
        return repr(self.weights)

    def apply(self, oper_list):
        r"""
        |a> = \sum_j w_j |j>
        In order to apply an operator, we use:
        O |a> = \sum_i |i> <i| O | a> = \sum_{i,j} w_j <i|O|j> |i>
        Each component of the new state is:
        |Oa>_i = \sum_j w_j <i|O|j> |i>
        Parameters
        ----------
        oper_list : TYPE
            DESCRIPTION.

        Returns
        -------
        a new LF_vector resulting from the application of the operator.

        """
        if isinstance(self.basis[0], basis_state):
            op_chain_gen = op_chain
            dot_prod_gen = dot_prod
        elif isinstance(self.basis[0], Liouville_basis_state):
            op_chain_gen = op_chain_LF
            dot_prod_gen = dot_prod_LF
        else:
            raise TypeError("Basis state must be of type basis_state" +
                            " or Liouville_basis_state")
        new_weights = np.zeros_like(self.weights, dtype=complex)
        for i in np.arange(self.n_states):
            for j in np.arange(self.n_states):
                O_j_state = op_chain_gen(self.basis[j], oper_list)
                i_O_j = dot_prod_gen(self.basis[i], O_j_state)
                new_weights[i] += self.weights[j] * i_O_j
        return LF_vector(self.basis, new_weights)


def create_Hubbard_basis_LF(n_sites, n_spins):
    """
    Function to create the set of basis vectors for the Hubbard model.
    Parameters
    ----------
    n_sites : int
        number of sites.
    n_spins : int
        number of spins.

    Returns
    -------
    states : list of Liouville_basis_states objects.
        List that contains all the basis states.

    """
    states = []
    for conf_f in product([0, 1], repeat=n_spins * n_sites):
        for conf_t in product([0, 1], repeat=n_spins * n_sites):
            conf_f = np.asarray(conf_f)
            conf_t = np.asarray(conf_t)
            occup_f = conf_f.reshape(n_sites, n_spins)
            occup_t = conf_t.reshape(n_sites, n_spins)
            state_in = Liouville_basis_state(n_sites, n_spins,
                                             occup_f, occup_t)
            # excl = False
            # for site in np.arange(2):
            #     if occup[site, 0] == 1 and occup[site, 1] == 1:
            #         excl = True
            #     if (occup == 0).all():
            #         excl = True
            # if excl:
            #     continue
            states.append(state_in)
    return states


if __name__ == "__main__":
    n_spins = 1
    n_sites = 1
    states = create_Hubbard_basis_LF(n_sites, n_spins)
    vec = LF_vector(states, np.ones((len(states),)))
    vec_1 = LF_vector(states, np.array([0.1, 0.2, 0.3, 0.4]))
    print(vec - vec_1)
    print(vec * vec_1)
    print(vec_1 * vec)
    print(2 * vec)
    print(- vec)
    site = 0
    oper_comb = ((cd_js_T, site, 0),
                 (c_js_T, site, 0),
                 (cd_js_F, site, 0),
                 (c_js_F, site, 0))
    print(vec_1)
    vec_2 = vec_1.apply(oper_comb)
    print(vec_1, vec_2)
    vec_3 = vec_1.apply([(c_js_F, site, 0)])
    print(vec_3)
    vec_3 = vec_1.apply([(cd_js_F, site, 0)])
    print(vec_3)
