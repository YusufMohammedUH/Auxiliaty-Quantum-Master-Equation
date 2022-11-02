from typing import Callable, Tuple
import numpy as np
from numba import njit, complex128, float64
import src.greens_function.frequency_greens_function as fg


@njit(complex128(float64, float64, float64, float64), cache=True)
def flat_bath_retarded(w: float, e0: float, D: float, gamma: float) -> complex:
    """Retarded Green's function with a flat band density of states.

    Parameters
    ----------
    w : float
        Frequency.

    e0 : float
        On-site potential

    D : float
        Half band width.

    gamma : float
        hight of the density of states

    Returns
    -------
    out: complex
        Retarded Green's function at frequency 'w'.
    """
    if gamma == 0:
        return 0. + 0.j
    x = D + (w - e0)
    y = (w - e0 - D)
    if x == 0:
        real_part = -np.inf
    elif y == 0:
        real_part = np.inf
    else:
        real_part = (-gamma) * np.log(np.abs(y / x))

    imag_part = 0
    if np.abs(w - e0) < D:
        imag_part = -np.pi * gamma
    return complex(real_part, imag_part)


@njit(complex128(float64, float64, float64, float64), cache=True)
def lorenzian_bath_retarded(w: float, e0: float, gamma: float,
                            v: float = 1.0) -> complex:
    """Retarded Green's function with a Lorenzian density of states.

    Parameters
    ----------
    w : float
        Frequency.

    e0 : float
        On-site potential

    D : float
        Half band width.

    gamma : float
        hight of the density of states

    Returns
    -------
    out: complex
        Retarded Green's function at frequency 'w'.
    """
    if v == 0:
        return 0. + 0.j
    x = w - e0 - 1j * gamma
    y = (w - e0)**2 + gamma**2
    result = 0
    if y == 0:
        result = complex(0, -np.inf)
    else:
        result = x / y
    return (v**2) * result


@njit(float64(float64, float64), cache=True)
def heaviside(x: float, x0: float = 0.0) -> float:
    """Heaviside function

    Parameters
    ----------
    x : float
        Running value.

    x0 : float
        Shift/offset.

    Returns
    -------
    out: float
        Value of the Heaviside function at 'x' with shift 'x0'.
    """
    if x < x0:
        return 0.
    elif x == 0:
        return 0.5
    else:
        return 1.


@njit(float64(float64, float64, float64, float64), cache=True)
def fermi(e: float, e0: float, mu: float, beta: float) -> float:
    """Fermi-Dirac distribution.


    Parameters
    ----------
    e : float
        Energie for which the fermi distribution is desired.

    e0 : float
        On-site potential.

    mu : float
        Chemical potential.

    beta : float
        Inverse Temperature times Boltzmann constant.

    Returns
    -------
    out: float
        Fermi-Dirac distribution for given parameters.
    """
    if beta == np.inf:
        x0 = e0 - mu
        return (1. - heaviside(e, x0))

    x = (e + e0 - mu) * beta
    if x < 0.0:
        return 1.0 / (1.0 + np.exp(x))
    else:
        return 1.0 - 1.0 / (1.0 + np.exp(-x))


# @njit(cache=True)
def _set_hybridization(freq: np.ndarray, retarded_function: Callable,
                       args: np.ndarray, keldysh_comp: str = "keldysh"
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Set the retarded and keldysh single particle Green's function from
    a supplied function determening the retarded function on a given frequency
    grid.

    Parameters
    ----------
    freq : numpy.ndarray (dim,)
        Frequency grid.

    retarded_function : function
        Function returning the retarded Green's function for given arguments.

    args: numpy.ndarray
        list of arguments necessary for calculating the Fermi function and
        the retarded Green's function.

    keldysh_comp: str
        Specify, which component out of keldysh, lesser or greater to be
        calculated additionally to the retarded component.

    Returns
    -------
    out: tuple (numpy.ndarray,numpy.ndarray)
        Tuple containing the retarded and keldysh/lesser/greater Green's
        functions.
    """
    e0 = args[0]
    mu = args[1]
    beta = args[2]
    retarded = np.array(
        [retarded_function(f, e0, args[3], args[4]) for f in freq])
    if keldysh_comp == "keldysh":
        keldysh = np.array([1.j * (1. / np.pi) * (
            1. - 2. * fermi(f, e0, mu, beta))
            * np.imag(retarded_function(f, e0, args[3], args[4]))
            for f in freq])

    elif keldysh_comp == "lesser":
        keldysh = np.array([-1.j * (1. / np.pi) * fermi(f, e0, mu, beta)
                            * np.imag(
                                retarded_function(f, e0, args[3], args[4]))
                            for f in freq])

    elif keldysh_comp == "greater":
        keldysh = np.array([1.j * (1. / np.pi) * (1. - fermi(f, e0, mu, beta))
                            * np.imag(
                                retarded_function(f, e0, args[3], args[4]))
                            for f in freq])
    return retarded, keldysh


def set_hybridization(freq: np.ndarray, retarded_function: Callable,
                      args: np.ndarray, keldysh_comp: str = "keldysh"
                      ) -> fg.FrequencyGreen:
    """Set the retarded and keldysh single particle Green's function from
    a supplied function determening the retarded function on a given frequency
    grid.

    Parameters
    ----------
    freq : numpy.ndarray (dim,)
        Frequency grid.

    retarded_function : function
        Function returning the retarded Green's function for given arguments.

    args: numpy.ndarray
        list of arguments necessary for calculating the Fermi function and
        the retarded Green's function.

    keldysh_comp: str
        Specify, which component out of keldysh, lesser or greater to be
        calculated additionally to the retarded component.

    Returns
    -------
    out: src.frequency_greens_function.FrequencyGreen
        Objectcontaining the calculated retarded and keldysh Green's functions.
    """
    retarded, keldysh = _set_hybridization(
        freq, retarded_function, args, keldysh_comp)
    return fg.FrequencyGreen(freq=freq, retarded=retarded, keldysh=keldysh,
                             keldysh_comp=keldysh_comp)
