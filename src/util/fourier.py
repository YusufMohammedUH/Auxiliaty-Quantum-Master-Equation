from typing import Union, Tuple
import numpy as np
import ctypes
from multiprocessing import Value, Array


def dft_point(fs: np.ndarray, ts: np.ndarray, w: float, sign: Union[-1, 1] = -1
              ) -> complex:
    """Preforms discrete Fourier transform of fs with corresponding domain ts
    at conjugated domain value w in both directions (sign=1 or sign=-1).

    Parameters
    ----------
    fs : np.ndarray
        Array to be transformed

    ts : np.ndarray
        Too fs corresponding time or frequency (domain)

    w : float
        Conjugated domain value, e.g a frequency, if ts is in time domain

    sign : Union[-1,1], optional
        Direction of transformation, by default -1

    Returns
    -------
    out: complex
        Fourier transformed fs from domain ts at w
    """
    if len(fs) != len(ts):
        raise ValueError("fs and ts must have same length")
    if len(fs) == 1:
        h = 1. + 0.j
    else:
        h = ts[1] - ts[0]

    if sign == 1:
        factor = h / (2 * np.pi)
    else:
        factor = h

    fs_w = np.array(fs) * np.exp(sign * 1j * w * np.array(ts))

    if len(fs_w) == 1:
        return factor * fs_w[0]
    else:
        return factor * (fs_w[1:-1].sum() + (fs_w[1] + fs_w[-1]) / 2.0)


def dft(fs: np.ndarray, ts: np.ndarray, ws: np.ndarray, sign: Union[-1, 1] = -1
        ) -> np.ndarray:
    """Preforms a discrete Fourier transform from given ts to ws.
    The Fourier transformed of fs for all values of ws is returned.

    Parameters
    ----------
    fs : np.ndarray
        Array to be transformed

    ts : np.ndarray
        Too fs corresponding time or frequency (domain)

    ws : np.ndarray
        Conjugated domain of fs where Fourier transform is wanted

    sign : Union[-1,1], optional
        Direction of transformation, by default -1

    Returns
    -------
    out: np.ndarray
        Fourier transformed fs from domain ts to domain ws
    """

    def dft_p(w): return dft_point(fs, ts, w, sign)
    dft_pv = np.vectorize(dft_p)
    return dft_pv(ws)


def dft_smooth(fs: np.ndarray, ts: np.ndarray, ws: np.ndarray,
               sigma: float = None, sign: Union[-1, 1] = -1) -> np.ndarray:
    """Preforms a discrete Fourier transform from given ts to ws. The function
    fs is multiplied with a gaussian function in order to smoothen the Fourier
    transform. The Fourier transformed of fs for all values of ws is returned.

    Parameters
    ----------
    fs : np.ndarray
        Array to be transformed

    ts : np.ndarray
        Too fs corresponding time or frequency (domain)

    ws : np.ndarray
        Conjugated domain of fs where Fourier transform is wanted

    sigma : _type_, optional
        Sets the with of a gaussian used to smoothen the function,
        by default None

    sign : Union[-1,1], optional
        Direction of transformation, by default -1

    Returns
    -------
    out: np.ndarray
        Fourier transformed fs from domain ts to domain ws
    """

    if sigma is None:
        sigma = np.sqrt((ts[-1] / 2)**2 - np.log(2))
    print(sigma)
    gaussian = np.exp(-(ts * ts) / (sigma**2))

    return dft(fs * gaussian, ts, ws, sign)


def init_shard_array_complex(shape: Tuple, data: Union[np.ndarray, None] = None
                             ) -> Array:
    """Given shape and data a complex shared memory array is created

    Parameters
    ----------
    shape : Tuple
        shape of data

    data : Union[np.ndarray, None], optional
        data of array to be shared, by default None

    Returns
    -------
    out:  Array
        shared array type with values of data if this is given
    """
    elements = 1
    for i in shape:
        elements *= i

    shared_array_base = Array(ctypes.c_double, elements * 2)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.view(np.complex128).reshape(shape)
    if data is not None:
        np.copyto(shared_array, data)
    return shared_array


def init_shard_array_real(shape: Tuple, data: Union[np.ndarray, None] = None
                          ) -> Array:
    """Given shape and data a real shared memory array is created

    Parameters
    ----------
    shape : Tuple
        shape of data

    data : Union[np.ndarray, None], optional
        data of array to be shared, by default None

    Returns
    -------
    out:  Array
        shared array type with values of data if this is given
    """

    elements = 1
    for i in shape:
        elements *= i

    shared_array_base = Array(ctypes.c_double, elements)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(shape)
    if data is not None:
        np.copyto(shared_array, data)
    return shared_array


def init_shard_value(value: Value, type_: str = 'd') -> Value:
    """Return a shared value = "value" and type= "type"

    Parameters
    ----------
    value : Value
        value to be shared

    type_ : str, optional
        type of value, by default 'd'

    Returns
    -------
    out: Value
        shared value
    """
    return Value(type_, value)


def shared_memory(gttp_: np.ndarray, t_: np.ndarray, tp_: np.ndarray,
                  ws_: np.ndarray) -> Tuple[Array, Array, Array, Array]:
    """Given attributes, converts these attributes to shared memory objects
       for parallel calculations

    Parameters
    ----------
    gttp_ : np.ndarray (dim, dim)
        Two time Green's function G(t,t') complex valued

    t_ : np.ndarray
        times t of gttp

    tp_ : np.ndarray
        times t' of gttp
    ws_ : np.ndarray
        frequencies to which gttp is transfromed

    Returns
    -------
    out: Tuple[Array, Array, Array, Array]
        tuple of arguments converted to shared memory arrays
    """
    gttp = init_shard_array_complex(gttp_.shape, gttp_)
    gtw = init_shard_array_complex((len(t_), len(ws_)))
    tp = init_shard_array_real(tp_.shape, tp_)
    ws = init_shard_array_real(ws_.shape, ws_)
    return (gttp, gtw, tp, ws)


def init_worker(gttp: np.ndarray, gtw: np.ndarray, tp: np.ndarray,
                ws: np.ndarray, h: float) -> None:
    """initializer of multiprocess pool function

    Parameters
    ----------
    gttp_ : np.ndarray (dim, dim)
        Two time Green's function G(t,t') complex valued

    gtw : np.ndarray (dim, dim)
         Green's function G(t,w) complex valued

    tp_ : np.ndarray
        times t' of gttp

    ws_ : np.ndarray
        frequencies to which gttp is transfromed

    h : float
        Times step size
    """
    gttp = gttp
    gtw = gtw
    tp = tp
    ws = ws
    h = h
