#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:49:15 2021

@author: Matteo Vandelli

Example script that shows how to evolve the 2-level system
using the Liouvillian decomposition and compares against Euler method.
"""
import numpy as np
import matplotlib.pyplot as plt
from liouville_decomposition import decompose_Liouville, evolve_Liouville


# Parameters describing the Rabi model
# epsilon0 is fixed to 0.
epsilon1 = 0.0
W = 1.1
Gamma = 0.7

# Liouvillian for 2-level system
Liouville = np.array([[0., 1j*W, -1j*W, Gamma],
                      [1j*W, 1j*epsilon1 - 0.5*Gamma, 0., -1j*W],
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
