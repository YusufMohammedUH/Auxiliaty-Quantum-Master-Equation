import numpy as np
import ctypes
from multiprocessing import Value, Array


def dft_point(fs, ts, w, sign=-1):
    """Preforms discrete Fourier transform on fs in both
       directions (sign=1 or sign=-1).

    Args:
        fs (list type complex elements): values to be transformed
        ts (list type float elements): corresponding time or frequency values
        w (float): freqency or time of transformed
        h (float): step size of ts
        sign (int (-1 or 1)): direction of transformation

    Returns:
        complex: value of transformed at w
    """
    if len(fs) != len(ts):
        raise ValueError("fs and ts must have same length")
    if len(fs) == 1:
        h = 1. + 0.j
    else:
        h = ts[1] - ts[0]

    if sign == 1:
        factor = h
    else:
        factor = (h / (2 * np.pi))

    fs_w = np.array(fs) * np.exp(sign * 1j * w * np.array(ts))

    if len(fs_w) == 1:
        return factor * fs_w[0]
    else:
        return factor * (fs_w[1:-1].sum() + (fs_w[1] + fs_w[-1]) / 2.0)


def get_ws(wmax, N):
    """Given wmax and N a numpy array is returned, where is a numpy array
       including wmax and 0.0

    Args:
        wmax (float): maximum value of list
        N (int): number of elements in the list

    Returns:
        np.array: numpy array of "frequency"
    """
    return np.array([-wmax + (2.0 * wmax * i) / N for i in range(N + 1)])


def dft(fs, ts, ws, sign=-1):
    """Preforms a discrete Fourier transform from given ts to ws.
    The Fourier transformed of fs for all values of ws is returned.

    Args:
        fs (list type complex elements): values to be transfromed
        ts (list type float elements): to fs corresponting time or frequency
                                       (domain)
        ws (list type float elements): conjugated domain of fs where Fourier
                                        transform is wanted
        sign (-1 or 1): direction of transformation

    Returns:
        numpy array with complex elements: Fourier transformed fs from domain
        ts to domain ws
    """
    def dft_p(w): return dft_point(fs, ts, w, sign)
    dft_pv = np.vectorize(dft_p)
    return dft_pv(ws)


def dft_smooth(fs, ts, ws, sigma=None, sign=-1):
    """Preforms a discrete Fourier transform from given ts to ws. The function
    fs is multiplied with a gaussian function in order to smoothen the Fourier
    transform. The Fourier transformed of fs for all values of ws is returned.

    Args:
        fs (list type complex elements): values to be transfromed
        ts (list type float elements): to fs corresponting time or frequency
                                       (domain)
        ws (list type float elements): conjugated domain of fs where Fourier
                                        transform is wanted
        h (float): step size of ts
        sigma ( = 10, float ): sets the with of a gaussian used to smoothen the
        function
        sign (-1 or 1): direction of transformation

    Returns:
        numpy array with complex elements: Fourier transformed fs from domain
        ts to domain ws
    """

    if sigma == None:
        sigma = np.sqrt((ts[-1] / 2)**2 - np.log(2))
    print(sigma)
    gaussian = np.exp(-(ts * ts) / (sigma**2))

    return dft(fs * gaussian, ts, ws, sign)


def init_shard_array_complex(shape, data=None):
    """Given shape and data a complex shared memory array is created

    Args:
        shape (tuple): shape of data
        data (numpy array, optional): data of array to be shared. Defaults to
        None.

    Returns:
        shared array: shared array type with values of data if this is given
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


def init_shard_array_real(shape, data=None):
    """Given shape and data a complex shared memory array is created

    Args:
        shape (tuple): shape of data
        data (numpy array, optional): data of array to be shared. Defaults to
        None.

    Returns:
        shared array: shared array type with values of data if this is given
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


def init_shard_value(value, type_='d'):
    """Return a shared value = "value" and type= "type"

    Args:
        value (type): value
        type_ (string discribing type of value, optional): type of value.
        Defaults to 'd'.

    Returns:
        shared value: shared value
    """
    return Value(type_, value)


def shared_memory(gttp_, t_, tp_, ws_):
    """Given attributes, converts these attributes to shared memory objects
       for parallel calculations

    Args:
        gttp_ (numpy array elements of complex): G(t,t')
        t_ (numpy array elements of double): time t
        tp_ (numpy array elements of double): time t'
        ws_ (numpy array elements of double): frequency w

    Returns:
        tuple: tuple of arguments converted to shared memory arrays
    """
    gttp = init_shard_array_complex(gttp_.shape, gttp_)
    gtw = init_shard_array_complex((len(t_), len(ws_)))
    tp = init_shard_array_real(tp_.shape, tp_)
    ws = init_shard_array_real(ws_.shape, ws_)
    return (gttp, gtw, tp, ws)


def init_worker(gttp, gtw, tp, ws, h):
    """initializer of multiprocess pool function

    Args:
        gttp (numpy array elements of complex): G(t,t')
        gtw (numpy array elements of complex): G(t,w)
        tp (numpy array elements of double): time t'
        ws (numpy array elements of double): frequency w
        h (float): step size of times t and t'
    """
    gttp = gttp
    gtw = gtw
    tp = tp
    ws = ws
    h = h
