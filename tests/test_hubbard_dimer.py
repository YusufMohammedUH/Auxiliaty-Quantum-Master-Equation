#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:22:52 2021

@author: Matteo Vandelli
"""
import numpy as np
import sys
sys.path.append("src")  # not pretty, fix it :(
from src import hubbard_model as hm


def test_generate_hubbard_hamil():
    """
    Test the correct generation of the Hubbard hamiltonian
    for an atom and a dimer.
    The test compares the eigenvalues against the analytical ones.
    """
    print("Generation of the Hubbard atom Hamiltonian")
    U = 2.0
    mu = 0.75
    Hamil, states = hm.Hubbard_atom(U, mu)
    print(Hamil)
    print("Hilbert space is ", states)
    # calculate the eigenvalues
    vals, vecs = np.linalg.eigh(Hamil)
    eig_vals_an = np.array([0., -mu, -mu, U-2*mu])
    print("Computed eigenvalues are:", np.sort(vals))
    assert np.allclose(np.sort(vals), np.sort(eig_vals_an), atol=1e-4)
    print("")
    print("Generation of the Hubbard dimer Hamiltonian")
    # Define hopping matrix
    t = np.array([[0.0, -0.2], [-0.2, 0.0]])
    mu = 0.0 * np.ones((2,))
    U_arr = U * np.ones((2,))
    n_sites = 2
    Hamil, states = hm.Hubbard_chain(t, U_arr, mu, n_sites)
    # print(Hamil)
    print("Hilbert space is ", states)
    vals, vecs = np.linalg.eigh(Hamil)
    # Analytical eigenvalues of the hubbard dimer
    eig_vals_an = np.array([0, 0, 0, 0, U, 2.*U,
                            t[0, 1], t[0, 1], -t[0, 1], -t[0, 1],
                            t[0, 1]+U, t[0, 1]+U, -t[0, 1]+U, -t[0, 1] + U,
                            0.5*U + np.sqrt((0.5*U)**2 + 4*t[0, 1]**2),
                            0.5*U - np.sqrt((0.5*U)**2 + 4*t[0, 1]**2)])
    print("Computed eigenvalues are:", np.sort(vals))
    assert np.allclose(np.sort(vals), np.sort(eig_vals_an), atol=1e-4)
