"""Calculate frequency dependent correlation function.
"""
from typing import Tuple, List, Union
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def get_two_point_correlator_frequency(green_component_plus: np.ndarray,
                                       green_component_minus: np.ndarray,
                                       freq: np.ndarray,
                                       precalc_correlators: List[
                                           np.ndarray],
                                       vals_sectors: List[np.ndarray],
                                       tensor_shapes: Tuple,
                                       permutation_sign: Union[
                                           Tuple[complex, complex],
                                           Tuple[None, None]] = (None, None),
                                       prefactor: Union[complex, None] = None,
                                       e_cut_off: Union[float, None] = None
                                       ) -> None:
    """Calculate the two point correlation function on mixed branches.

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of expectation value, e.g. -1j for single particle green's
        function and -1 for susceptibility

    e_cut_off : float
        Cut off eigenvalue. This drops the eigenvalue of the steady state
        density of states, so only if the eigenvalues correspond to the (0,0)
        sector. This effects the susceptibility only in the 'ch' and 'z'
        channels.

    """

    for i in prange(len(freq)):
        G_plus = 0 + 0j
        G_minus = 0 + 0j

        for n in prange(tensor_shapes[0][0]):
            L_n = vals_sectors[0][n]
            if np.abs(L_n) > e_cut_off:

                G_plus -= (precalc_correlators[0][n] /
                           ((1j * freq[i] + L_n)))

        for n in prange(tensor_shapes[1][0]):
            L_n = vals_sectors[1][n]
            if np.abs(L_n) > e_cut_off:
                G_minus += (precalc_correlators[1][n] /
                            ((1j * freq[i] - L_n)))

        green_component_plus[i] = prefactor * permutation_sign[0] * G_plus
        green_component_minus[i] = prefactor * permutation_sign[1] * G_minus


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_mmm(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (---)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j
            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]

                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * precalc_correlators[0][n, m] \
                                * (1.0 / ((1j * w1 + L_n) *
                                          (1j * (w1 + w2) + L_n + L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]

                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2]\
                                * precalc_correlators[1][n, m] * (-1 / (
                                    (1j * w1 + L_n) * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]

                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * precalc_correlators[2][n, m] * (1 / (
                                    (1j * (w1 + w2) + L_n + L_m)
                                    * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]

                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1]\
                                * precalc_correlators[3][n, m] * (-1 / (
                                    (1j * w1 - L_n - L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[4][1]):
                        L_m = vals_sectors[4][1][m]

                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[1] \
                                * permutation_sign[2]\
                                * precalc_correlators[4][n, m] * (1 / (
                                    (1j * (w1 + w2) - L_n)
                                    * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[5][1]):
                        L_m = vals_sectors[5][1][m]

                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[1] \
                                * permutation_sign[2] \
                                * permutation_sign[0] \
                                * precalc_correlators[5][n, m]\
                                * (1 / ((1j * w1 - L_n - L_m)
                                        * (1j * (w1 + w2) - L_n))
                                   )

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_pmm(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (+--)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * precalc_correlators[0][n, m] \
                                * (1.0 / (
                                    (1j * w1 + L_n) * (1j * w2 + L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * precalc_correlators[1][n, m] \
                                * (-1 / (
                                    (1j * w1 - L_n - L_m) * (1j * w2 + L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2]\
                                * precalc_correlators[2][n, m] * (-1 / (
                                    (1j * w1 + L_n) * (1j * (w1 + w2) - L_m)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2]\
                                * precalc_correlators[3][n, m] * (1 / (
                                    (1j * w1 - L_n - L_m)
                                    * (1j * (w1 + w2) - L_n)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_mpm(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (-+-)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * precalc_correlators[0][n, m] * (1.0 / (
                                    (1j * w1 + L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * precalc_correlators[1][n, m] * (-1 / (
                                    (1j * w1 + L_n) * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * permutation_sign[1]\
                                * precalc_correlators[2][n, m] * (-1 / (
                                    (1j * (w1 + w2) - L_m) * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * permutation_sign[1]\
                                * precalc_correlators[3][n, m] * (1 / (
                                    (1j * w2 - L_n - L_m)
                                    * (1j * (w1 + w2) - L_n)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_mmp(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (--+)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[1]\
                                * permutation_sign[2]\
                                * precalc_correlators[0][n, m] * (-1 / (
                                    (1j * w1 + L_m) * (1j * (w1 + w2) - L_n)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[1]\
                                * permutation_sign[2]\
                                * precalc_correlators[1][n, m] * (1 / (
                                    (1j * w1 + L_n)
                                    * (1j * (w1 + w2) + L_n + L_m)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[1]\
                                * permutation_sign[2] * permutation_sign[0]\
                                * precalc_correlators[2][n, m] * (-1 / (
                                    (1j * w2 + L_m) * (1j * (w1 + w2) - L_n)))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[1]\
                                * permutation_sign[2] * permutation_sign[0]\
                                * precalc_correlators[3][n, m] * (1 / (
                                    (1j * w2 + L_n)
                                    * (1j * (w1 + w2) + L_n + L_m)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_ppm(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (++-)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * precalc_correlators[0][n, m] \
                                * (1.0 / ((1j * (w1 + w2) + L_n + L_m)
                                          * (1j * w2 + L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * precalc_correlators[1][n, m] \
                                * (-1 / ((1j * w1 - L_n - L_m)
                                         * (1j * w2 + L_n)))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * precalc_correlators[2][n, m]\
                                * ((1 / (1j * w1 - L_n - L_m))
                                   - (1 / (1j * (w1 + w2) - L_m))
                                   ) * (1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * precalc_correlators[3][n, m]\
                                * ((1 / (1j * w1 + L_m))
                                   - (1 / (1j * (w1 + w2) + L_n + L_m))
                                   ) * (1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[4][1]):
                        L_m = vals_sectors[4][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * precalc_correlators[4][n, m] * (-1 / (
                                    (1j * (w1 + w2) - L_m)
                                    * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[5][1]):
                        L_m = vals_sectors[5][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * precalc_correlators[5][n, m] * (1 / (
                                    (1j * (w1 + w2) - L_m)
                                    * (1j * w2 - L_n - L_m)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_pmp(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (+-+)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2]\
                                * precalc_correlators[0][n, m] * (1 / (
                                    (1j * (w1 + w2) - L_m)
                                    * (1j * w2 - L_n - L_m)))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2]\
                                * precalc_correlators[1][n, m]\
                                * ((1 / (1j * w1 - L_n - L_m))
                                   - (1 / (1j * (w1 + w2) - L_n))
                                   ) * (1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2]\
                                * precalc_correlators[2][n, m]\
                                * (1 / (1j * w1 - L_n - L_m)) \
                                * (-1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2] \
                                * permutation_sign[1]\
                                * precalc_correlators[3][n, m]\
                                * (-1 / (1j * w1 + L_m)) \
                                * (1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[4][1]):
                        L_m = vals_sectors[4][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2] \
                                * permutation_sign[1]\
                                * precalc_correlators[4][n, m]\
                                * ((1 / (1j * w1 + L_n))
                                   - (1 / (1j * (w1 + w2) + L_n + L_m))
                                   ) * (1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[5][1]):
                        L_m = vals_sectors[5][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2] \
                                * permutation_sign[1]\
                                * precalc_correlators[5][n, m] * (1 / (
                                    (1j * (w1 + w2) + L_n + L_m)
                                    * (1j * w2 + L_m)))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_mpp(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (-++)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1] \
                                * precalc_correlators[0][n, m]\
                                * ((1 / (1j * w1 - L_n - L_m))
                                   - (1 / (1j * (w1 + w2) - L_m))
                                   ) * (1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1]\
                                * precalc_correlators[1][n, m]\
                                * (1 / (1j * (w1 + w2) - L_n)) * (
                                    1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1]\
                                * precalc_correlators[2][n, m]\
                                * (-1 / (1j * w1 + L_m)) \
                                * (1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1]\
                                * permutation_sign[2] \
                                * precalc_correlators[3][n, m]\
                                * (1 / (1j * w1 - L_n - L_m))\
                                * (-1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[4][1]):
                        L_m = vals_sectors[4][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1]\
                                * permutation_sign[2] \
                                * precalc_correlators[4][n, m]\
                                * (1 / (1j * (w1 + w2) + L_n + L_m)) * (
                                    1 / (1j * w2 + L_n))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[5][1]):
                        L_m = vals_sectors[5][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1]\
                                * permutation_sign[2] \
                                * precalc_correlators[5][n, m]\
                                * ((1 / (1j * w1 + L_m)) - (
                                    1 / (1j * (w1 + w2) + L_n + L_m))
                                   ) * (1 / (1j * w2 + L_n))

            green_component[i, j] = G


@njit(parallel=True, cache=True)
def get_three_point_correlator_frequency_ppp(green_component: np.ndarray,
                                             freq: np.ndarray,
                                             precalc_correlators: List[
                                                 np.ndarray],
                                             vals_sectors: List[np.ndarray],
                                             tensor_shapes: Tuple,
                                             permutation_sign: Tuple,
                                             prefactor: complex,
                                             e_cut_off: float):
    """Calculate the three point correlation function component (+++)
    from parameters

    Parameters
    ----------
    green_component_plus : np.ndarray
        Component for positive times

    green_component_minus : np.ndarray
        Component for negative times

    freq : np.ndarray
        1D frequency grid

    precalc_correlators : List[np.ndarray]
        Precalculated expectation value of operators at t=0

    vals_sectors : List[np.ndarray]
        List of eigenvalues

    tensor_shapes : Tuple
        Tuple of total number of eigenvalues in each sector (diagonal)

    permutation_sign : complex
        Permutation sign

    prefactor : complex
        Prefactor of the correlation function
    """

    for i in prange(len(freq)):
        w1 = freq[i]
        for j in prange(len(freq)):
            G = 0 + 0j

            w2 = freq[j]
            for n in range(tensor_shapes[0][0]):
                L_n = vals_sectors[0][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in range(tensor_shapes[0][1]):
                        L_m = vals_sectors[0][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor\
                                * precalc_correlators[0][n, m]\
                                * (1 / (1j * w1 - L_n - L_m)) * (
                                    1 / (1j * (w1 + w2) - L_m))

            for n in prange(tensor_shapes[1][0]):
                L_n = vals_sectors[1][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[1][1]):
                        L_m = vals_sectors[1][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[2]\
                                * precalc_correlators[1][n, m]\
                                * (1 / (1j * w1 - L_n - L_m)) \
                                * (-1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[2][0]):
                L_n = vals_sectors[2][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[2][1]):
                        L_m = vals_sectors[2][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0]\
                                * precalc_correlators[2][n, m]\
                                * (1 / (1j * (w1 + w2) - L_m)) * (
                                    1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[3][0]):
                L_n = vals_sectors[3][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[3][1]):
                        L_m = vals_sectors[3][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1] \
                                * precalc_correlators[3][n, m]\
                                * (-1 / (1j * w1 + L_m)) * (
                                    -1 / (1j * w2 - L_n - L_m))

            for n in prange(tensor_shapes[4][0]):
                L_n = vals_sectors[4][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[4][1]):
                        L_m = vals_sectors[4][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[1] \
                                * permutation_sign[2]\
                                * precalc_correlators[4][n, m]\
                                * (1 / (1j * (w1 + w2) + L_n + L_m)) * (
                                    1 / (1j * w2 + L_m))

            for n in prange(tensor_shapes[5][0]):
                L_n = vals_sectors[5][0][n]
                if np.abs(L_n) > e_cut_off:
                    for m in prange(tensor_shapes[5][1]):
                        L_m = vals_sectors[5][1][m]
                        if np.abs(L_m) > e_cut_off:
                            G += prefactor * permutation_sign[0] \
                                * permutation_sign[1]\
                                * permutation_sign[2] \
                                * precalc_correlators[5][n, m]\
                                * (1 / (1j * w1 + L_m)) * (
                                    1 / (1j * (w1 + w2) + L_n + L_m))

            green_component[i, j] = G
