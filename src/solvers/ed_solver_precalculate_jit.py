"""Calculate expectation values at time t=0. These can be calculated and saved
to be used later for calculating time dependent or frequency dependent green's
functions
"""
from typing import List, Tuple
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def precalculate_two_point_correlator(shape: Tuple, left_vacuum: np.ndarray,
                                      spin_sector_fermi_ops: List[np.ndarray],
                                      vec_r_sector: List[np.ndarray],
                                      vec_l_sector: List[np.ndarray],
                                      rho_stready_state: np.ndarray,
                                      n: int) -> np.ndarray:
    """Calculate two point correlation function at t=0 for a set of operators
    given by spin_sector_ops

    Parameters
    ----------
    shape : Tuple
        shape of resulting Matrix/Tensor
    left_vacuum : np.ndarray
        Left Vacuum state (see SuperFermionicOperators)
    spin_sector_fermi_ops : List[np.ndarray]
        List of operators in spin sector
    vec_r_sector : List[np.ndarray]
        List of right eigenvectors in spin sector
    vec_l_sector : List[np.ndarray]
        List of left eigenvectors in spin sector
    rho_stready_state : np.ndarray
        Steady state density operator (see SuperFermionicOperators)
    n : int
        number of operators in expectation value/spin_sector_fermi_ops

    Returns
    -------
    np.ndarray
        Two point correlation function at t=0 for a set of operators
    given by spin_sector_ops
    """
    assert n == 2
    precalc_corr_tmp = np.zeros(shape, dtype=np.complex128)
    tmp = 0. + 0.j
    for i in prange(shape[0]):
        tmp = left_vacuum.dot(spin_sector_fermi_ops[0]).dot(
            vec_r_sector[0][i]).dot(
            vec_l_sector[0][i]).dot(
            spin_sector_fermi_ops[1]).dot(
            rho_stready_state)[0, 0]
        if np.abs(tmp) > 1e-30:
            precalc_corr_tmp[i] = tmp

    return precalc_corr_tmp


@njit(parallel=True, cache=True)
def precalculate_three_point_correlator(shape: Tuple, left_vacuum: np.ndarray,
                                        spin_sector_fermi_ops: List[
                                            np.ndarray],
                                        vec_r_sector: List[np.ndarray],
                                        vec_l_sector: List[np.ndarray],
                                        rho_stready_state: np.ndarray,
                                        n: int) -> np.ndarray:
    """Calculate three point correlation function at t=0 for a set of operators
    given by spin_sector_ops

    Parameters
    ----------
    shape : Tuple
        shape of resulting Matrix/Tensor
    left_vacuum : np.ndarray
        Left Vacuum state (see SuperFermionicOperators)
    spin_sector_fermi_ops : List[np.ndarray]
        List of operators in spin sector
    vec_r_sector : List[np.ndarray]
        List of right eigenvectors in spin sector
    vec_l_sector : List[np.ndarray]
        List of left eigenvectors in spin sector
    rho_stready_state : np.ndarray
        Steady state density operator (see SuperFermionicOperators)
    n : int
        number of operators in expectation value/spin_sector_fermi_ops

    Returns
    -------
    np.ndarray
        Three point correlation function at t=0 for a set of operators
    given by spin_sector_ops
    """
    assert n == 3
    precalc_corr_tmp = np.zeros(shape, dtype=np.complex128)

    expectation_start = left_vacuum.dot(spin_sector_fermi_ops[0])
    tmp = 0. + 0.j
    for i in prange(shape[0]):
        expectation_val = expectation_start.dot(
            vec_r_sector[0][i]).dot(
            vec_l_sector[0][i]).dot(
            spin_sector_fermi_ops[1])

        for j in prange(shape[1]):
            tmp = expectation_val.dot(
                vec_r_sector[1][j]).dot(
                vec_l_sector[1][j]).dot(
                spin_sector_fermi_ops[2]).dot(
                rho_stready_state)[0, 0]
            if np.abs(tmp) > 1e-30:
                precalc_corr_tmp[i, j] = tmp
    return precalc_corr_tmp


@njit(parallel=True, cache=True)
def precalculate_four_point_correlator(shape: Tuple, left_vacuum: np.ndarray,
                                       spin_sector_fermi_ops: List[np.ndarray],
                                       vec_r_sector: List[np.ndarray],
                                       vec_l_sector: List[np.ndarray],
                                       rho_stready_state: np.ndarray,
                                       n: int) -> np.ndarray:
    """Calculate four point correlation function at t=0 for a set of operators
    given by spin_sector_ops

    Parameters
    ----------
    shape : Tuple
        shape of resulting Matrix/Tensor
    left_vacuum : np.ndarray
        Left Vacuum state (see SuperFermionicOperators)
    spin_sector_fermi_ops : List[np.ndarray]
        List of operators in spin sector
    vec_r_sector : List[np.ndarray]
        List of right eigenvectors in spin sector
    vec_l_sector : List[np.ndarray]
        List of left eigenvectors in spin sector
    rho_stready_state : np.ndarray
        Steady state density operator (see SuperFermionicOperators)
    n : int
        number of operators in expectation value/spin_sector_fermi_ops

    Returns
    -------
    np.ndarray
        Four point correlation function at t=0 for a set of operators
    given by spin_sector_ops
    """
    assert n == 4
    precalc_corr_tmp = np.zeros(shape, dtype=np.complex128)

    expectation_start = left_vacuum.dot(spin_sector_fermi_ops[0])

    for i in prange(shape[0]):
        expectation_val_1 = expectation_start.dot(
            vec_r_sector[0][i]).dot(
            vec_l_sector[0][i]).dot(
            spin_sector_fermi_ops[1])

        for j in prange(shape[1]):
            expectation_val_2 = expectation_val_1.dot(
                vec_r_sector[1][j]).dot(
                vec_l_sector[1][j]).dot(
                spin_sector_fermi_ops[2])
            for k in prange(shape[2]):
                precalc_corr_tmp[i, j, k] = expectation_val_2.dot(
                    vec_r_sector[2][k]).dot(
                    vec_l_sector[2][k]).dot(
                    spin_sector_fermi_ops[3]).dot(
                    rho_stready_state)[0, 0]

    return precalc_corr_tmp
