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
    occup = np.array([[1, 0],    # first site
                      [1, 1]])   # second site
    state_1 = fs.state(2, 2, occup)
    occup[1, 1] -= 1
    state_2 = fs.state(2, 2, occup)
    print(state_2, state_1)
    dot_prod_11 = fs.dot_prod(state_1, state_1)
    dot_prod_12 = fs.dot_prod(state_1, state_2)
    assert dot_prod_11 == 1. and dot_prod_12 == 0.
    

