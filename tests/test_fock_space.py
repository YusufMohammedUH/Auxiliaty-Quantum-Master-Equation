#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 23:13:34 2021

@author: Matteo Vandelli
"""
import numpy as np
from src import define_states_fock as fs


def test_kron_delta():
    assert fs.kron_delta(2, 2) == 1 and fs.kron_delta(2, 3) == 0


def test_state_creation():
    # spinful state with 2 sites
    target_state = "|10>|11>"
    occup = np.array([[1, 0],    # first site
                      [1, 1]])   # second site
    built_state = fs.state(2, 2, occup)
    assert target_state == repr(built_state)


def test_dot_prod():
    r"""
    Test dot product on basis states.
    We create two spinless sites in the Fock state: |10>|11>
    Test that:
    (<01|<11|) \dot (|01>|11>) = 1.
    (<01|<10|) \dot (|01>|11>) = 0.
    (<01|<1-1|) \dot (|01>|1-1>) = 0. (product of null vectors is 0)
    """
    occup = np.array([[1, 0],    # first site
                      [1, 1]])   # second site
    state_1 = fs.state(2, 2, occup)
    occup[1, 1] -= 1
    state_2 = fs.state(2, 2, occup)
    occup[1, 1] -= 1
    state_3 = fs.state(2, 2, occup)
    dot_prod_11 = fs.dot_prod(state_1, state_1)
    dot_prod_12 = fs.dot_prod(state_1, state_2)
    dot_prod_33 = fs.dot_prod(state_3, state_3)
    assert dot_prod_11 == 1. and dot_prod_12 == 0. and dot_prod_33 == 0.


def test_c_js():
    """
    Test annihilation operator acting on a state.
    We create two spinless sites in the Fock state: |1>|0>
    c_{0}|1>|0> = |0>|0>
    c_{1}|1>|0> = |0>|-1>  (-1 is the null state)
    """
    occup = np.array([[1],    # first site
                      [0]])   # second site
    state_1 = fs.state(2, 1, occup)
    state_2 = fs.c_js(0, 0, state_1)
    state_3 = fs.c_js(1, 0, state_1)
    test1 = (repr(state_1) == "|1>|0>")
    test2 = (repr(state_2) == "|0>|0>")
    test3 = (repr(state_3) == "|1>|-1>")
    assert test1 and test2 and test3


def test_cd_js():
    """
    Test creation operator acting on a state.
    We create two spinless sites in the Fock state: |1>|0>
    cd_{0}|1>|0> = |-1>|0>   (-1 is the null state)
    cd_{1}|1>|0> = |1>|1>
    """
    occup = np.array([[1],    # first site
                      [0]])   # second site
    state_1 = fs.state(2, 1, occup)
    state_2 = fs.cd_js(0, 0, state_1)
    state_3 = fs.cd_js(1, 0, state_1)
    test1 = (repr(state_1) == "|1>|0>")
    test2 = (repr(state_2) == "|-1>|0>")
    test3 = (repr(state_3) == "|1>|1>")
    assert test1 and test2 and test3


def test_operator_chain_density():
    """
    Test chain of operators acting on a state.
    We create two spinless sites in the Fock state:
    Number operator on state |0>|1>:
    <0|<1|cd_0 c_0 |0>|1> = 0
    <0|<1|cd_1 c_1 |0>|1> = 1
    """
    occup = np.array([[0],    # first site
                      [1]])   # second site
    state_1 = fs.state(2, 1, occup)
    state_2 = fs.op_chain(state_1, ((fs.cd_js, 0, 0), (fs.c_js, 0, 0)))
    state_3 = fs.op_chain(state_1, ((fs.cd_js, 1, 0), (fs.c_js, 1, 0)))
    test1 = (fs.dot_prod(state_1, state_2) == 0.)
    test2 = (fs.dot_prod(state_1, state_3) == 1.)
    assert test1 and test2


def test_operator_chain_ddoubleocc():
    """
    Test chain of operators acting on a state.
    We create two spinless sites in the Fock state.
    Double occupancy operator of sites on |1>|1> and |0>|1>:
    <0|<1|cd_0 c_0 cd_1 c_1 |0>|1> = 0
    <1|<1|cd_0 c_0 cd_1 c_1 |1>|1> = 1
    """
    occup = np.array([[0],    # first site
                      [1]])   # second site
    state_in1 = fs.state(2, 1, occup)
    occup = np.array([[1],    # first site
                      [1]])   # second site
    state_in2 = fs.state(2, 1, occup)
    state_out1 = fs.op_chain(state_in1, ((fs.cd_js, 0, 0), (fs.c_js, 0, 0),
                                         (fs.cd_js, 1, 0), (fs.c_js, 1, 0)))
    state_out2 = fs.op_chain(state_in2, ((fs.cd_js, 0, 0), (fs.c_js, 0, 0),
                                         (fs.cd_js, 1, 0), (fs.c_js, 1, 0)))
    test1 = (fs.dot_prod(state_in1, state_out1) == 0.)
    test2 = (fs.dot_prod(state_in2, state_out2) == 1.)
    assert test1 and test2
