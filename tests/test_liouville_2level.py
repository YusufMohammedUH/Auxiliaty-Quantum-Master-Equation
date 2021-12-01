#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:20:21 2021

@author: Matteo Vandelli
"""
import numpy as np
import sys
sys.path.append("src")  # not pretty, fix it :(
from src import application_2level_sys as lv2


def test_liouvillian_2level():
    W = 1.
    epsilon0 = 0.
    epsilon1 = 0.3
    Gamma = 0.2
    Liouvillian = lv2.create_Liouvillian_2level(W, epsilon0, epsilon1, Gamma)
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
