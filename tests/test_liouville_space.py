#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:25:26 2021

@author: Matteo Vandelli
"""

import numpy as np
import sys
sys.path.append("src")  # not pretty, fix it :(
from src import define_states_liouville as lfs


def test_state_creation_LF():
    """
    Test creation of a state in Fock-Liouville space.
    spinful state with 2 sites, symmetric in Tilde and Fock spaces.
    """
    target_state = "|10>|11> x |10>|11>"
    occup = np.array([[1, 0],    # first site
                      [1, 1]])   # second site
    occupt = np.array([[1, 0],    # first site
                      [1, 1]])   # second site
    built_state = lfs.Liouville_basis_state(2, 2, occup, occupt)
    assert target_state == repr(built_state)


def test_dot_prod_LF():
    r"""
    Test dot product on basis states.
    We create two spinless sites in the Liouville-Fock state: |1>|1> x |0>|0>
    Test several cases, namely:
    (<1|<1| x <0|<0|) \dot (|1>|1> x |0>|0>) = 1.
    (<1|<1| x <0|<0|) \dot (|1>|1> x |0>|1>) = 0.
    (<1|<1| x <0|<-1|) \dot (|1>|1> x |0>|-1>) = 0.
    """
    occup = np.array([[1],    # first site
                      [1]])   # second site
    occupt = np.array([[0],    # first site
                      [0]])   # second site
    state_1 = lfs.Liouville_basis_state(2, 1, occup, occupt)
    occupt[1, 0] += 1
    state_2 = lfs.Liouville_basis_state(2, 1, occup, occupt)
    occupt[1, 0] -= 2
    state_3 = lfs.Liouville_basis_state(2, 1, occup, occupt)
    dot_prod_11 = lfs.dot_prod_LF(state_1, state_1)
    dot_prod_12 = lfs.dot_prod_LF(state_1, state_2)
    dot_prod_33 = lfs.dot_prod_LF(state_3, state_3)
    assert (dot_prod_11 == 1.) and (dot_prod_12 == 0.) and (dot_prod_33 == 0.)


def test_c_js_LF():
    r"""
    Test annihilation operator acting on a state.
    We create two spinless sites in the Liouville-Fock state: |1>|0> x |0>|1>
    Apply c_0, \tilde{c}_0, c_1, \tilde{c}_1.
    """
    occup = np.array([[1],    # first site
                      [0]])   # second site
    occupt = np.array([[0],    # first site
                      [1]])   # second site
    state_1 = lfs.Liouville_basis_state(2, 1, occup, occupt)
    state_2 = lfs.c_js_T(0, 0, state_1)
    state_3 = lfs.c_js_T(1, 0, state_1)
    state_4 = lfs.c_js_F(0, 0, state_1)
    state_5 = lfs.c_js_F(1, 0, state_1)
    test1 = (repr(state_1) == "|1>|0> x |0>|1>")
    test2 = (repr(state_2) == "|1>|0> x |-1>|1>")
    test3 = (repr(state_3) == "|1>|0> x |0>|0>")
    test4 = (repr(state_4) == "|0>|0> x |0>|1>")
    test5 = (repr(state_5) == "|1>|-1> x |0>|1>")
    assert test1 and test2 and test3 and test4 and test5


def test_cd_js_LF():
    r"""
    Test creation operator acting on a state.
    We create two spinless sites in the Liouville-Fock state: |1>|0> x |0>|1>
    Apply c_0, \tilde{c}_0, c_1, \tilde{c}_1.
    """
    occup = np.array([[1],    # first site
                      [0]])   # second site
    occupt = np.array([[0],    # first site
                      [1]])   # second site
    state_1 = lfs.Liouville_basis_state(2, 1, occup, occupt)
    state_2 = lfs.cd_js_T(0, 0, state_1)
    state_3 = lfs.cd_js_T(1, 0, state_1)
    state_4 = lfs.cd_js_F(0, 0, state_1)
    state_5 = lfs.cd_js_F(1, 0, state_1)
    test1 = (repr(state_1) == "|1>|0> x |0>|1>")
    test2 = (repr(state_2) == "|1>|0> x |1>|1>")
    test3 = (repr(state_3) == "|1>|0> x |0>|-1>")
    test4 = (repr(state_4) == "|-1>|0> x |0>|1>")
    test5 = (repr(state_5) == "|1>|1> x |0>|1>")
    assert test1 and test2 and test3 and test4 and test5


def test_operator_chain_density_LF():
    """
    Test chain of operators acting on a state.
    We create two spinless sites in the Fock state:
    Number operator on state |LF> = |0>|1> x |1>|0>:
    <LF| cd_0_F c_0_F |LF> = 0
    <LF| cd_0_T c_0_T |LF>= 1
    <LF| cd_1_F c_1_F |LF> = 1
    <LF| cd_1_T c_1_T |LF> = 0
    """
    occup = np.array([[0],    # first site
                      [1]])   # second site
    occupt = np.array([[1],    # first site
                      [0]])   # second site
    state_1 = lfs.Liouville_basis_state(2, 1, occup, occupt)
    state_2 = lfs.op_chain_LF(state_1, ((lfs.cd_js_F, 0, 0),
                                        (lfs.c_js_F, 0, 0)))
    state_3 = lfs.op_chain_LF(state_1, ((lfs.cd_js_T, 0, 0),
                                        (lfs.c_js_T, 0, 0)))
    state_4 = lfs.op_chain_LF(state_1, ((lfs.cd_js_F, 1, 0),
                                        (lfs.c_js_F, 1, 0)))
    state_5 = lfs.op_chain_LF(state_1, ((lfs.cd_js_T, 1, 0),
                                        (lfs.c_js_T, 1, 0)))
    test1 = (lfs.dot_prod_LF(state_1, state_2) == 0.)
    test2 = (lfs.dot_prod_LF(state_1, state_3) == 1.)
    test3 = (lfs.dot_prod_LF(state_1, state_4) == 1.)
    test4 = (lfs.dot_prod_LF(state_1, state_5) == 0.)
    assert test1 and test2 and test3 and test4
