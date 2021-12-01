#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:31:12 2021

@author: Matteo Vandelli

Construct the super Liouvillian for a 2 level system
"""
import numpy as np
from scipy.linalg import eig


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
