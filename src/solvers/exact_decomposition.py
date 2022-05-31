from typing import Tuple
import numpy as np
from scipy import linalg
from numba import njit


@njit(cache=True)
def _exact_spectral_decomposition(vals: np.ndarray, vec_r_tmp: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray]:
    """Buy jit optimized function returning the eigenvalue and
    eigenvectors.


    Parameters
    ----------
    vals : np.ndarray (dim,)
        Eigenvalues.

    vec_r_tmp : np.ndarray (dim,dim)
        Right eigenvectors. The ith eigenvector is given by vec_r_tmp[:, i].

    Returns
    -------
    (vals, vec_l, vec_r) : tuple ((dim,),(dim, 1, dim),(dim, dim, 1))
        Eigen decomposition of the, in general non-hermitian, complex matrix.
        The first element contains the eigenvalues, the second the
        left and the third the right eigenvectors. The ith eigenvector, e.g.
        left is given by vec_l[i]. The eigenvectors are numpy.matrix, for
        convenient.
    """
    dim = vals.shape[0]
    vec_l_tmp = np.linalg.inv(vec_r_tmp.T)
    vec_l = np.zeros((dim, 1, dim), dtype=np.complex64)
    vec_r = np.zeros((dim, dim, 1), dtype=np.complex64)

    for i in range(dim):
        vec_r[i] = vec_r_tmp[:, i].reshape((dim, 1))
        vec_l[i] = vec_l_tmp[:, i].reshape((1, dim))

    return vals, vec_l, vec_r


def exact_spectral_decomposition(matrix: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray]:
    """Exact diagonalization of the matrix, setting the attributes
    vals, containing the eigen values, vec_l and vec_r containing the left
    and right eigen vectors respectively.

    Parameters
    ----------

    matrix : numpy.ndarray (dim,dim)
        General non-hermitian, complex matrix
    Returns
    -------
    (vals, vec_l, vec_r) : tuple ((dim,),(dim, 1, dim),(dim, dim, 1))
        Eigen decomposition of the, in general non-hermitian, complex matrix.
        The first element contains the eigenvalues, the second the
        left and the third the right eigenvectors
    """

    vals, vec_r_tmp = linalg.eig(matrix)

    return _exact_spectral_decomposition(vals, vec_r_tmp)


@njit(cache=True)
def time_propagation_all_times_exact_diagonalization(times: np.ndarray,
                                                     vec0: np.ndarray,
                                                     vals: np.ndarray,
                                                     vec_l: np.ndarray,
                                                     vec_r: np.ndarray
                                                     ) -> np.ndarray:
    """Retruns the time propagated vector "vec0" propagated by supplying
    eigenvectors ("vec_l","vec_r"), eigenvalues "vals" for all times "times".

    Parameters
    ----------
    times : numpy.ndarray (dim,)
        array of times for which the propagated vector is to be calculated

    vec0 : numpy.ndarray (dim,1)
        Initial vector.

    vals : numpy.ndarray (dim,)
        Eigenvalues.

    vec_l: numpy.ndarray (dim,1,dim)
        Left eigenvectors.

    vec_r: numpy.ndarray (dim,1)
        Right eigenvectors.

    Returns
    -------
    vec : numpy.ndarray (dim, dim2)
        Time propagated vectors. vec[:,i] contains the time
        propagated vector at time times[i]
    """
    dim = vec0.shape[0]
    vec = np.zeros((dim,) + times.shape, dtype=np.complex64)
    dim_times = times.shape[0]
    for l1 in np.arange(dim):
        vec += np.exp(vals[l1] * times)[:].reshape((1, dim_times))\
            * vec_r[l1] * ((vec_l[l1]).dot(vec0))
    return vec


@njit(cache=True)
def time_propagation_exact_diagonalization(time: np.ndarray,
                                           vec0: np.ndarray,
                                           vals: np.ndarray,
                                           vec_l: np.ndarray,
                                           vec_r: np.ndarray
                                           ) -> np.ndarray:
    """Returns the time propagated vector "vec0" propagated by supplying
    eigenvectors ("vec_l","vec_r"), eigenvalues "vals" for time "time".


    Parameters
    ----------
    time : float
        time for which the propagated vector is to be calculated

    vec0 : numpy.ndarray (dim,1)
        Initial vector.

    vals : numpy.ndarray (dim,)
        Eigenvalues.

    vec_l: numpy.ndarray (dim,1,dim)
        Left eigenvectors.

    vec_r: numpy.ndarray (dim,1)
        Right eigenvectors.

    Returns
    -------
    vec : numpy.ndarray (dim, dim2)
        Time propagated vectors. vec[:,i] contains the time
        propagated vector at time times[i]
    """
    dim = vec0.shape[0]
    vec = np.zeros((dim, 1), dtype=np.complex64)
    for l1 in np.arange(dim):
        vec += np.exp(vals[l1] * time) * vec_r[l1] * (
            (vec_l[l1]).dot(vec0))
    return vec


@njit(cache=True)
def time_evolution_operator(time: float, vals: np.ndarray, vec_l: np.ndarray,
                            vec_r: np.ndarray) -> np.ndarray:
    """Retruns the time evolution operator for given time "time", by
    using the supplied eigen decomposition.

    Parameters
    ----------
    times : float
        time for which the propagated vector is to be calculated

    vals : numpy.ndarray (dim,)
        Eigenvalues.

    vec_l: numpy.ndarray (dim,1,dim)
        Left eigenvectors.

    vec_r: numpy.ndarray (dim,1)
        Right eigenvectors.

    Returns
    -------
    out : numpy.ndarray (dim, dim)
        Time propagation operator, for given time, from
        eigen decomposition supplied by as function arguments.
    """

    dim = vals.shape[0]
    time_evolution_operator = np.zeros((dim, dim), dtype=np.complex64)
    for l1 in np.arange(dim):
        time_evolution_operator += np.exp(vals[l1]
                                          * time) * vec_r[l1] * vec_l[l1]
    return time_evolution_operator
