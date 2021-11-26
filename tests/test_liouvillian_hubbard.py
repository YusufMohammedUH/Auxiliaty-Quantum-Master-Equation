#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:07:07 2021

@author: Matteo Vandelli
"""
import numpy as np
# import matplotlib.pyplot as plt
from src.liouvillian_hubbard import naive_Liouvillian_Hubbard, Dissip_Naka_LF
from src.liouville_decomposition import decompose_Liouville


def test_liouville_Naka_Dissip():
    """
    Calculation of the Liouvillian of the damped Hubbard atom,
    introduced in Nakagawa paper.
    To achieve this, we disable the coupling to leads (Gamma1 and Gamma2)
    by putting them to 0 and we introduce the Dissip_Naka_LF,
    i.e. a simple 2-particle dissipation.
    """
    # Single atom parameters
    t = np.array([[0]])  # np.array([[0.0, -0.2], [-0.2, 0.0]])
    mu = 1.0 * np.ones((1,))
    U_arr = 2. * np.ones((1,))
    # we define Gamma as the coefficient of the 2-particle dissipator
    Gamma = 0.2 * np.ones((1,))
    n_sites = 1
    # Exact eigenvalues of the damped Hubbard atom
    # see Nakagawa et al. PhysRevLett.126.110404
    eig_vals_an = np.array([0., - mu[0], - mu[0],
                            U_arr[0] - 1j * Gamma[0] - 2*mu[0]])
    # Construct Liouvillian of the model
    Liouville, states = naive_Liouvillian_Hubbard(
        t, 0, 0, U_arr, mu, n_sites)
    # construct dissipator of the model
    Dissip = Dissip_Naka_LF(states, Gamma, n_sites)
    Liouville += Dissip
    print("States : ", states)
    vals, vecs_l, vecs_r = decompose_Liouville(Liouville)
    vals = np.sort(vals)
    print("Eigenvalues :", vals)

    eig_liouv_an = np.matrix.flatten(
        eig_vals_an[:, None] - np.conj(eig_vals_an[None, :]))
    print("Eigenvalues exact: ", np.sort(-1j*eig_liouv_an))
    assert np.allclose(np.sort(-1j*eig_liouv_an), vals)
    # plt.figure()
    # plt.title("Eigenvalues of the Liouvillian")
    # for val in vals:
    #     plt.scatter(vals.real, vals.imag)
    # plt.xlabel(r"$Re \, \Lambda$")
    # plt.ylabel(r"$Im \, \Lambda$")
    # plt.show()
