#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:31:12 2021

@author: Matteo Vandelli

Construct the super Liouvillian for a 2 level system
"""
import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt


def decompose_Liouville(Liouville):
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
    vals, vec_l, vec_r = eig(Liouville, left=True)
    # scipy only checks that vec_l.conj().T @ vec_t = diagonal_matrix
    # because of this, the order in vec_l and vec_r differs sometimes.
    # have to do the following to ensure generalized orthogonality of
    # eigenvalues in degenerate subspaces
    vec_l_ = np.linalg.inv(vec_r.T)
    return vals, vec_l_, vec_r


def evolve_Liouville(Liouville, rho0, times):
    """
    Contrutct the Liouvillian at different times, starting from an initial
    density matrix (in vector form) rho0.

    Parameters
    ----------
    Liouville : numpy.ndarray (dim, dim)
        Matrix in Fock-Liouville space
    rho0 : numpy.array (dim,)
        Initial density matrix in Fock-Liouville space
    times : numpy.array (n_times,)
        array with the times at which the Liouvillian should be computed.
    Returns
    -------
    rho : numpy.ndarray (n_times, dim)
        Density matrix in Fock-Liouville space at the different time steps.
    """
    vals, vec_l, vec_r = decompose_Liouville(Liouville)
    dim = rho0.shape[0]
    rho = np.zeros((dim,)+times.shape, dtype=complex)
    for l1 in np.arange(dim):
        rho += np.exp(vals[l1] * times)[None, :] * vec_r[:, None, l1] * (
                (vec_l[:, l1]).dot(rho0))
    return rho


if __name__ == "__main__":
    # Parameters describing the Rabi model
    # epsilon0 is fixed to 0.
    epsilon1 = 0.0
    W = 1.1
    Gamma = 0.7

    # Liouvillian for 2-level system
    Liouville = np.array([[0., 1j*W, -1j*W, Gamma],
                          [1j*W, -1j*epsilon1 - 0.5*Gamma, 0., -1j*W],
                          [-1j*W, 0, -1j*epsilon1 - 0.5*Gamma, 1j*W],
                          [0., -1j*W, 1j*W, -Gamma]
                          ])

    # Plotting the eigenvalues of the Liouvillian
    vals, vec_l, vec_r = decompose_Liouville(Liouville)

    plt.figure()
    plt.title("Eigenvalues of the Liouvillian operator")
    for val in vals:
        plt.scatter(val.real, val.imag)
    plt.ylabel(r"${\rm Im}\,\Lambda_\alpha$")
    plt.xlabel(r"${\rm Re}\,\Lambda_\alpha$")

    # initial condition
    rho0 = np.array([0., 0., 0., 1.])

    # Time evolution parameters
    t0 = 0
    tf = 10
    N = 1000
    times = np.linspace(t0, tf, N)
    dt = times[1]-times[0]

    rho_eu = rho0
    rho00_eu = []
    rho01_eu = []
    rho10_eu = []
    rho11_eu = []

    # Evolution with Euler formula
    for t in times:
        rho_eu = rho_eu + Liouville.dot(rho_eu)*dt
        rho00_eu.append(rho_eu[0])
        rho01_eu.append(rho_eu[1])
        rho10_eu.append(rho_eu[2])
        rho11_eu.append(rho_eu[3])

    # Evolution with Liouvillian decomposition
    rho = evolve_Liouville(Liouville, rho0, times)

    # Plotting comparison between methods
    plt.figure()
    plt.plot(times[::20], rho[0, ::20].real, 'bo', label=r"$\rho_{00}$ diag")
    plt.plot(times[::20], rho[3, ::20].real, 'ro', label=r"$\rho_{11}$ diag")
    plt.plot(times, np.asarray(rho00_eu).real, 'b--',
             label=r"$\rho_{00}$ Euler")
    plt.plot(times, np.asarray(rho11_eu).real, 'r--',
             label=r"$\rho_{11}$ Euler")
    plt.ylabel("Density Matrix")
    plt.xlabel("Time")
    plt.legend()
    plt.show()
