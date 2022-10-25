from typing import Union
import numpy as np
from . import convert_keldysh_components as conv
from . import keldysh_identity as kid
import src.util.hdf5_util as hd5


class TimeGreen:
    """Simple time Green's function container. Contains the
    single particle Green's on the Keldysh contour in Keldysh rotated
    representation, e.g. G^R and G^K. G^A is dropt do to symmetry [G^R]* = G^A.

    Parameters
    ----------
    time : numpy.ndarry (dim,)
        1-D time grid

    retarded : numpy.ndarry (dim,), optional
        Contains the retarded Green's, by default None

    keldysh : numpy.ndarry (dim,), optional
        Contains the keldysh/lesser/greater Green's, by default None


    Attributes
    ----------
    time : numpy.ndarry (dim,)
        1-D time grid

    retarded : numpy.ndarry (dim,)
        Contains the retarded Green's, by default None

    keldysh : numpy.ndarry (dim,)
        Contains the keldysh/lesser/greater Green's, by default None


    Raises
    ------
    TypeError
        "time must be of type numpy.array!"

    TypeError
        "retarded must be of type numpy.array or None!"

    TypeError
        "keldysh must be of type numpy.array or None!"

    ValueError
        "time and retarded must have same shape"

    ValueError
        "time and keldysh must have same shape"
    """

    def __init__(self, time: np.ndarray,
                 retarded: Union[np.ndarray, None] = None,
                 keldysh: Union[np.ndarray, None] = None) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        if not isinstance(time, np.ndarray):
            raise TypeError("ERROR: time must be of type numpy.array!")
        if (not isinstance(retarded, np.ndarray)) and (retarded is not None):
            raise TypeError(
                "ERROR: retarded must be of type numpy.array or None!")
        if (not isinstance(keldysh, np.ndarray)) and (keldysh is not None):
            raise TypeError(
                "ERROR: keldysh must be of type numpy.array or None!")

        self.time = time
        if self.time.flags.writeable:
            self.time.flags.writeable = False

        if retarded is None:
            self.retarded = np.zeros(len(time), dtype=np.complex128)
        else:
            if time.shape != retarded.shape:
                raise ValueError(
                    "ERROR: time and retarded must have same shape")
            self.retarded = np.copy(retarded)
        if keldysh is None:
            self.keldysh = np.zeros(len(time), dtype=np.complex128)
        else:
            if time.shape != keldysh.shape:
                raise ValueError(
                    "ERROR: time and keldysh must have same shape")
            self.keldysh = np.copy(keldysh)

    def copy(self) -> "TimeGreen":
        """Return a copy of the object.

        Returns
        -------
        out: TimeGreen
        """
        return TimeGreen(self.freq, self.retarded.copy(),
                         self.keldysh.copy())

    def __add__(self, other: Union["TimeGreen", "kid.KeldyshIdentity",
                                   int, float, complex]) -> "TimeGreen":
        """Add TimeGreen or KeldyshIdentity objects or number from
        current object and return resulting TimeGreen.

        In case of a number 'other' and Green's function 'G' and the identity
        matrix 'I' in the Keldysh contour

        other*I+G

        is returned

        Parameters
        ----------
        other : TimeGreen

        Returns
        -------
        out: TimeGreen
        """
        if isinstance(other, TimeGreen):
            return TimeGreen(self.time, self.retarded + other.retarded,
                             self.keldysh + other.keldysh)
        elif (isinstance(other, int) or isinstance(other, float)
              or isinstance(other, complex)):
            return TimeGreen(self.time, self.retarded + other,
                             self.keldysh)
        elif isinstance(other, kid.KeldyshIdentity):
            return other + self

    def __sub__(self, other: Union["TimeGreen", "kid.KeldyshIdentity",
                                   int, float, complex]) -> "TimeGreen":
        """Subtract TimeGreen or KeldyshIdentity objects or number from
        current object and return resulting TimeGreen.

        In case of a number 'other' and Green's function 'G' and the identity
        matrix 'I' in the Keldysh contour

        other*I-G

        is returned

        Parameters
        ----------
        other : TimeGreen

        Returns
        -------
        out: TimeGreen
        """
        if isinstance(other, TimeGreen):
            return TimeGreen(self.time, self.retarded - other.retarded,
                             self.keldysh - other.keldysh)
        elif (isinstance(other, int) or isinstance(other, float)
              or isinstance(other, complex)):
            return TimeGreen(self.time, self.retarded - other,
                             self.keldysh)
        elif isinstance(other, kid.KeldyshIdentity):
            return (other - self) * (-1)

    def __mul__(self, other: Union["TimeGreen", "kid.KeldyshIdentity",
                                   int, float, complex]) -> "TimeGreen":
        """Multiply two time Green's functions.

        The resulting Green's function is obtained by the
        corresponding Langreth rules:

        `c = a*b`

        with

        `c.retarded = a.retarded*b.lesser+a.lesser*b.advanced`

        and

        `c.keldysh = a.lesser*b.greater+a.greater*b.lesser`

        Parameters
        ----------
        other : TimeGreen

        Returns
        -------
        out: TimeGreen
        """

        if isinstance(other, TimeGreen):
            # obtaining lesser and greater components
            a_greater = conv.get_greater_from_keldysh(self)
            a_lesser = conv.get_lesser_from_keldysh(self)
            b_greater = conv.get_greater_from_keldysh(other)
            b_lesser = conv.get_lesser_from_keldysh(other)
            c_keldysh = a_greater * b_lesser + a_lesser * b_greater
            return TimeGreen(time=self.time, retarded=(
                self.retarded * other.retarded), keldysh=(
                c_keldysh))
        elif (isinstance(other, int) or isinstance(other, float)
              or isinstance(other, complex)):
            return TimeGreen(time=self.time, retarded=self.retarded * other,
                             keldysh=self.keldysh * other)
        elif isinstance(other, kid.KeldyshIdentity):
            # the keldysh_identity as non-zero entries only in the
            # main diagonal, therefore the lesser,greater components are zero
            # the langreth rules for multiplication lead to vanishing retarded
            # advanced and keldysh components
            return TimeGreen(time=self.time)

    def save(self, fname: str, dir_: str, dataname: str,
             savefreq: bool = True) -> None:
        """Save the Green's function to a file.

        Parameters
        ----------
        fname : str
            File name of the hdf5 file.
        dir : str
            Groupe name/subdirectory within the hdf5 file, e.g. '/' or
            '/system'.
        dataname : str
            Name under which the green's function should be saved.
        """
        hd5.add_data(fname, f"{dir_}/{dataname}", 'keldysh', self.keldysh)
        hd5.add_data(fname, f"{dir_}/{dataname}", 'retarded', self.retarded)
        if savefreq:
            hd5.add_attrs(fname, f"{dir}/{dataname}",
                          {"time_min": self.freq[0], "time_max": self.freq[-1],
                           'N_time': len(self.time)})

    def load(self, fname: str, dir_: str, dataname: str,
             readfreq: bool = True) -> None:
        """Load data from hdf5 file to Green's function object

        Parameters
        ----------
        fname : str
            File name of the hdf5 file

        dir : str
            Directory/group in which Green's is stored

        dataname : str
            Name of the Green's function
        """
        attrs = hd5.read_attrs(fname, f"{dir_}/{dataname}")

        if readfreq:
            if (attrs['time_max'] != self.time[-1] or
                attrs['time_min'] != self.time[0]
                    or attrs['N_time'] != self.time.shape[0]):
                raise ValueError("Time grid of loaded data doesn't" +
                                 " match object time grid.")

        self.retarded = hd5.read_data(file=fname,
                                      dir_=f"{dir_}/{dataname}",
                                      dataname='retarded')
        self.keldysh = hd5.read_data(file=fname, dir_=f"{dir_}/{dataname}",
                                     dataname='keldysh')
