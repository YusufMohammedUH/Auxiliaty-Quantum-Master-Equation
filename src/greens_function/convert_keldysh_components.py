from typing import Union
import numpy as np
import src.greens_function.frequency_greens_function as fg
import src.greens_function.time_greens_function as tg


def get_keldysh_from_lesser(green: Union["fg.FrequencyGreen", 'tg.TimeGreen']
                            ) -> np.ndarray:
    """Return the keldysh component from the lesser and retarded component
    supplied by the Green's function green.

    Parameters
    ----------
    green : Union[fg.FrequencyGreen, tg.TimeGreen]
        Green's function in time or frequency domain

    Returns
    -------
    np.ndarray
        Keldysh component
    """
    keldysh = 1j * (1 / np.pi) * green.retarded.imag + 2 * green.keldysh
    return keldysh


def get_lesser_from_keldysh(green: Union["fg.FrequencyGreen", 'tg.TimeGreen']
                            ) -> np.ndarray:
    """Return the lesser component from the keldysh and retarded component
    supplied by the Green's function green

    Parameters
    ----------
    green : Union[fg.FrequencyGreen, tg.TimeGreen]
        Green's function in time or frequency domain

    Returns
    -------
    np.ndarray
        Lesser component
    """
    lesser = 0.5 * (green.keldysh - 1j * (1 / np.pi) * green.retarded.imag)
    return lesser


def get_greater_from_keldysh(green: Union["fg.FrequencyGreen", 'tg.TimeGreen']
                             ) -> np.ndarray:
    """Return the greater component from the keldysh and retarded component
    supplied by the Green's function green

    Parameters
    ----------
    green : Union[fg.FrequencyGreen, tg.TimeGreen]
        Green's function in time or frequency domain

    Returns
    -------
    np.ndarray
        Greater component
    """
    greater = green.keldysh - get_lesser_from_keldysh(green)
    return greater
