#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:59:42 2021

@author: Matteo Vandelli
"""
import sys
import numpy as np
from scipy.linalg import eig
from liouvillian_hubbard import naive_Liouvillian_Hubbard, Dissip_Dorda_LF
from define_states_liouville import cd_js_F, c_js_F
from state_vector import LF_vector


def decompose_Liouville_states(Liouville, basis):
    """
    Find eigenvalues and left/right eigenvectors of the Liouvillian.

    Parameters
    ----------
    Liouville : numpy.ndarray (dim, dim)
        Matrix in Fock-Liouville space
    Returns
    -------
    (vals, vec_l, vec_r) : tuple with
                           - vals : numpy.array with eigenvalues
                           - vec_l : left eigenvectors as
                                     defined by scipy.linalg.eig
                           - vec_r : right eigenvectors as
                                     defined by scipy.linalg.eig
    """
    vals, vec_l_, vec_r = eig(Liouville, left=True)
    # scipy only checks that vec_l.conj().T @ vec_t = diagonal_matrix
    # because of this, the order in vec_l and vec_r differs sometimes.
    # have to do the following to ensure generalized orthogonality of
    # eigenvalues in degenerate subspaces
    vec_l = np.conj(np.linalg.inv(vec_r.T))
    states_r = []
    states_l = []
    for ind in np.arange(vals.shape[0]):
        state_r = LF_vector(basis, vec_r[:, ind])
        state_l = LF_vector(basis, vec_l[:, ind])
        states_r.append(state_r)
        states_l.append(state_l)
    return vals, states_l, states_r


def find_nearest(a, a0):
    """Element in nd array `a` closest to the scalar value `a0`,
    returns the index"""
    idx = np.abs(a - a0).argmin()
    return idx


def green_function_Liouvillian(omega, Liouville_sl,
                               states, component=(0, 0, 0, 0, ">"),
                               delta=1e-6):
    """
    Calculate the Green's function using the Liouvillian.
    Parameters
    ----------
    omega : numpy.array (n_freqs,)
        Array with the desired frequencies at which
        the GF should be computed.
    Liouville_sl : numpy.ndarray (n_states, n_states)
        Liouvillian in Liouville-Fock space.
    states : list of type Liouvillian_basis_state
        Basis of the Liouville-Fock space.
    component : tuple/list, optional
        Component of the Green's function.
        It has to contain (site1, site2, band1, band2, keldysh component).
        The default is (0, 0, 0, 0, ">").
    delta : float, optional
        Artificial broadening. The default is 1e-6.

    Raises
    ------
    ValueError
        If the Liovillian has more than 1 state very close to 0.

    Returns
    -------
    Gf : numpy.ndarray (n_freqs,)
        DESCRIPTION.

    """
    b1, b2, sp1, sp2, cmp_keldysh = component
    vals, vecs_l, vecs_r = decompose_Liouville_states(Liouville_sl, states)
    # assert that there is only one steady state
    idx = find_nearest(vals, 0.0)
    if not np.allclose(vals[idx], 0.0):
        print("Error: Liovillian has more than 1 steady state!")
        sys.exit()
    rho_stat_r = vecs_r[idx]
    rho_stat_l = vecs_l[idx]
    Gf = np.zeros_like(omega, dtype=complex)
    # construct Gf = <L_0| c_0 |R> <L| c^\dagger_0 | R_0>/(omega - i*\Lambda)
    for ind in np.arange(vals.shape[0]):
        R_s = vecs_r[ind]
        L_s = vecs_l[ind]
        I_c_R = rho_stat_l * R_s.apply([[c_js_F, b1, sp1]])
        L_cd_rho = L_s * rho_stat_r.apply([[cd_js_F, b2, sp2]])
        # print(L_cd_rho)
        Gf += I_c_R * (L_cd_rho) / (omega + 1j*delta - 1j*vals[ind])
    if cmp_keldysh == ">":
        # constrcut G_lesser
        Gf -= np.conj(Gf)
    elif cmp_keldysh == "<":
        Gf -= np.conj(Gf)
        Gf = np.conj(Gf)
    else:
        raise ValueError("Unknown gf component.")
    # TODO implemtent other components
    return Gf


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = 0.0 * np.array([[1.]])
    mu = 0.0 * np.array([[1.]])
    U = 2.0 * np.array([[1.]])
    Gamma1 = 0.05 * np.array([[1.]])
    Gamma2 = 0.05 * np.array([[1.]])
    Liouville_sl, states = naive_Liouvillian_Hubbard(
        t, (Gamma1, Gamma2), U, mu, 1, n_spins=2, Dissip=Dissip_Dorda_LF)
    print("States : ", states)
    vals, vecs_l, vecs_r = decompose_Liouville_states(Liouville_sl, states)
    print("Eigenvalues :", np.sort(vals))
    plt.figure()
    plt.title("Eigenvalues of the Liouvillian")
    for val in vals:
        plt.scatter(vals.real, vals.imag)
    plt.xlabel(r"$Re \, \Lambda$")
    plt.ylabel(r"$Im \, \Lambda$")
    plt.show()
    # compute Green's function
    omega = np.linspace(-4, 4, 1000)
    Gf = green_function_Liouvillian(omega, Liouville_sl, states)
    Gf -= green_function_Liouvillian(omega, Liouville_sl,
                                     states, component=(0, 0, 0, 0, "<"))
    plt.figure()
    plt.plot(omega, (0.5j/np.pi*Gf).real)
    plt.xlabel("Energy")
    plt.ylabel("Spectral function")
    plt.show()
