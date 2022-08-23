from typing import Tuple, Union
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import src.auxiliary_mapping.auxiliary_system_parameter as auxp
import src.util.hdf5_util as hd5


@njit(cache=True)
def _z_aux(ws: np.ndarray, N: int, Nb: int, E: np.ndarray, Gamma1: np.ndarray,
           Gamma2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the non-interacting auxiliary greens function at the impurity
    site in frequency domain.

    Parameters
    ----------
    ws : numpy.ndarray (dim,)
        Frequency grid.

    N : int
        Number of sites in the auxiliary system.

    Nb : int
        Number of bath sites to the left/right
            (total number of bath sites is 2*Nb)

    E : numpy.ndarray (N, N)
        T-Matrix (hopping and onsite potential)
        of the Auxiliary Model

    Gamma1 : numpy.ndarray (N, N)
        Coupling to Markovian bath 1

    Gamma2 : numpy.ndarray (N, N)
        Coupling to Markovian bath 2

    Returns
    -------
    G_retarded, G_keldysh: tuple ((dim,),(dim,))
        Tuple containing the non-interacting retarded and keldysh single
        particle greens function in frequency domain.
    """
    id = np.identity(N)
    Z_R_aux = np.zeros((ws.shape[0], N, N), dtype=np.complex128)
    Z_A_aux = np.zeros((ws.shape[0], N, N), dtype=np.complex128)
    Z_K_aux = np.zeros((ws.shape[0], N, N), dtype=np.complex128)
    for i, w in enumerate(ws):
        Z_R_aux[i] = np.linalg.inv(
            w * id - E + 1.0j * (Gamma2 + Gamma1))
        Z_A_aux[i] = np.linalg.inv(w * id - E -
                                   1.0j * (Gamma2 + Gamma1))
        Z_K_aux[i] = 2.0j * \
            (Z_R_aux[i].dot((Gamma2 - Gamma1).dot(Z_A_aux[i])))
    return (Z_R_aux[:, Nb, Nb],
            Z_K_aux[:, Nb, Nb])


@njit(cache=True)
def _get_self_enerqy(green_0_ret_inverse: np.ndarray, green_ret: np.ndarray,
                     green_kel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the self-energy from the supplied retarded and keldysh
    greens function.

    Parameters
    ----------
    green_0_ret_inverse : numpy.ndarray (dim,)
        Non-interacting, uncoupled, inverse, retarded Green's function.
        Is equal to the freqency grid, in absence of a chemical potential and
        onsite potential.

    green_ret : numpy.ndarray (dim,)
        Full retarded Green's function.

    green_kel : numpy.ndarray (dim,)
        Full keldysh Green's function.

    Returns
    -------
    Sigma_retarded, Sigma_keldysh: tuple ((dim,),(dim,))
        Tuple containing the retarded and keldysh single particle self-energy
        in frequency domain.
    """
    green_ret_w_inverse = np.zeros(
        green_kel.shape, dtype=np.complex128)
    sigma_ret_w = np.zeros(
        green_kel.shape, dtype=np.complex128)
    sigma_kel_w = np.zeros(
        green_kel.shape, dtype=np.complex128)
    for i in range(green_0_ret_inverse.shape[0]):
        if green_ret[i] == 0:
            green_ret_w_inverse[i] = np.inf
        else:
            green_ret_w_inverse[i] = 1.0 / green_ret[i]
        sigma_ret_w[i] = green_0_ret_inverse[i] - \
            green_ret_w_inverse[i]
        green_ret_w_abs_square_inverse = 0
        if (green_ret[i] * np.conj(green_ret[i])) == 0:
            green_ret_w_abs_square_inverse = np.inf
        else:
            green_ret_w_abs_square_inverse = (
                1 / (green_ret[i] * np.conj(green_ret[i])))
        sigma_kel_w[i] = (
            green_ret_w_abs_square_inverse * green_kel[i])
    return sigma_ret_w, sigma_kel_w


class FrequencyGreen:
    """Simple frequency Green's function container.

    Parameters
    ----------
        freq : numpy.ndarry (dim,)
            1-D Frequency grid

        retarded : numpy.ndarry (dim,)
            Contains the retarded Green's

        keldysh : numpy.ndarry (dim,)
            Contains the keldysh Green's

    Attributes
    ----------
        freq : numpy.ndarry (dim,)
            1-D Frequency grid

        retarded : numpy.ndarry (dim,)
            Contains the retarded Green's

        keldysh : numpy.ndarry (dim,)
            Contains the keldysh Green's
    """

    def __init__(self, freq: np.ndarray,
                 retarded: Union[np.ndarray, None] = None,
                 keldysh: Union[np.ndarray, None] = None) -> None:
        if not isinstance(freq, np.ndarray):
            raise TypeError("freq must be of type numpy.array!")
        if (not isinstance(retarded, np.ndarray)) and (retarded is not None):
            raise TypeError("retarded must be of type numpy.array or None!")
        if (not isinstance(keldysh, np.ndarray)) and (keldysh is not None):
            raise TypeError("keldysh must be of type numpy.array or None!")

        self.freq = freq
        if self.freq.flags.writeable:
            self.freq.flags.writeable = False
        if retarded is None:
            self.retarded = np.zeros(len(freq), dtype=np.complex128)
        else:
            if freq.shape != retarded.shape:
                raise ValueError("freq and retarded must have same shape")
            self.retarded = np.copy(retarded)
        if keldysh is None:
            self.keldysh = np.zeros(len(freq), dtype=np.complex128)
        else:
            if freq.shape != keldysh.shape:
                raise ValueError("freq and keldysh must have same shape")
            self.keldysh = np.copy(keldysh)

    def copy(self) -> "FrequencyGreen":
        """Return a copy of the object.

        Returns
        -------
        out: FrequencyGreen
        """
        return FrequencyGreen(self.freq, self.retarded.copy(),
                              self.keldysh.copy())

    def __add__(self, other: "FrequencyGreen") -> "FrequencyGreen":
        """Add two FrequencyGreen objects

        Parameters
        ----------
        other : FrequencyGreen

        Returns
        -------
        out: FrequencyGreen
        """
        return FrequencyGreen(self.freq, self.retarded + other.retarded,
                              self.keldysh + other.keldysh)

    def __sub__(self, other: "FrequencyGreen") -> "FrequencyGreen":
        """Add two FrequencyGreen objects

        Parameters
        ----------
        other : FrequencyGreen

        Returns
        -------
        out: FrequencyGreen
        """
        return FrequencyGreen(self.freq, self.retarded - other.retarded,
                              self.keldysh - other.keldysh)

    def __mul__(self, other: "FrequencyGreen") -> "FrequencyGreen":
        """Multiply two frequency Green's functions.

        A multiplication in frequency domain corresponds to a convolution in
        time, therefore the resulting Green's function is obtained by the
        corresponding Langreth rules:

        `c = a*b`

        with

        `c.retarded = a.retarded*b.retarded`

        and

        `c.retarded = a.retarded*b.keldysh+a.keldysh*b.retarded`

        Parameters
        ----------
        other : FrequencyGreen

        Returns
        -------
        out: FrequencyGreen
        """
        if isinstance(other, FrequencyGreen):
            return FrequencyGreen(self.freq, self.retarded * other.retarded,
                                  self.retarded * other.keldysh +
                                  self.keldysh * other.retarded.conj())
        elif (isinstance(other, int) or isinstance(other, float)
              or isinstance(other, complex)):
            return FrequencyGreen(self.freq, self.retarded * other,
                                  self.retarded +
                                  self.keldysh)

    def inverse(self) -> "FrequencyGreen":
        """Return the inverse Green's function

        Returns
        -------
        out: FrequencyGreen
            Inverse Green's function
        """
        retarded = 1. / self.retarded
        keldysh = -retarded * self.keldysh * retarded.conj()
        return FrequencyGreen(self.freq, retarded=keldysh)

    def dyson(self, green_0_ret_inverse: np.ndarray,
              self_energy: "FrequencyGreen") -> None:
        """Calculate and set the the frequency Green's function, through the
        Dyson equation for given self-energy sigma.

        Parameters
        ----------
        green_0_ret_inverse : numpy.array (dim,)
            Inverse isolated, non-interacting Green's function
            :math:`[g^R_0(w)]^{-1}`

        self_energy : FrequencyGreen
            Self-energy could be only the hybridization/embeding self-energy
            or include the interacting self-energy
        """
        self.retarded = 1.0 / (green_0_ret_inverse - self_energy.retarded)
        self.keldysh = (self.retarded * self_energy.keldysh *
                        self.retarded.conj())

    def set_green_from_auxiliary(self, auxsys: auxp.AuxiliarySystem) -> None:
        """Set the retarded and Keldysh impurity site Green's function
        (at auxsys.Nb), by supplied AuxiliarySystem object.

        Parameters
        ----------
        auxsys : auxiliary_system_parameter.AuxiliarySystem
            Auxiliary system parameters class
        """
        assert np.array_equal(auxsys.ws, self.freq)

        self.retarded, self.keldysh = _z_aux(auxsys.ws, auxsys.N, auxsys.Nb,
                                             auxsys.E, auxsys.Gamma1,
                                             auxsys.Gamma2)

    def get_self_enerqy(self,
                        green_0_ret_inverse: Union[np.ndarray, None] = None
                        ) -> "FrequencyGreen":
        """Returns the retarded and Keldysh component of the self-energy
        from the retarded and Keldysh Green's.

        In case of no interacting self-energy the resulting self-energy is
        the embeding self-energy or hybridization function.

        Parameters
        ----------
        green_0_ret_inverse : numpy.array (dim,)
            Inverse isolated, non-interacting Green's function
            :math:`[g^R_0(w)]^{-1}`

        Returns
        -------
        out : FrequencyGreen
            self-energy of given Green's functions
        """
        if green_0_ret_inverse is None:
            green_0_ret_inverse = self.freq

        sigma = _get_self_enerqy(
            green_0_ret_inverse, self.retarded, self.keldysh)

        singularities = np.where(sigma[0].real == float("-inf"))[0]
        for s in singularities:
            if s == 0:
                sigma[0][s] = complex(
                    np.sign(sigma[0].real[s + 1]) *
                    np.abs(np.real(sigma[0][s])),
                    np.imag(sigma[0][s]))
                sigma[1][s] = 0
            else:
                sigma[0][s] = complex(
                    np.sign(sigma[0].real[s - 1]) *
                    np.abs(np.real(sigma[0][s])),
                    np.imag(sigma[0][s]))
                sigma[1][s] = 0

        return FrequencyGreen(self.freq, sigma[0], sigma[1])

    def save(self, fname: str, dir: str, dataname: str,
             savefreq: bool = False) -> None:
        """Save the Green's function to a file.

        Parameters
        ----------
        fname : str
            File name of the hdf5 file.
        dir : str
            Groupe name/subdirectory within the hdf5 file, e.g. '/' or '/green'.
        dataname : str
            Name of dataset/ of the green's function.
        """
        hd5.add_data(fname, f"{dir}/{dataname}", 'keldysh', self.keldysh)
        hd5.add_data(fname, f"{dir}/{dataname}", 'retarded', self.retarded)
        if savefreq:
            hd5.add_attrs(fname, f"{dir}/{dataname}",
                          {"freq_min": self.freq[0], "freq_max": self.freq[-1],
                           'N_freq': len(self.freq)})


def get_hyb_from_aux(auxsys: auxp.AuxiliarySystem) -> "FrequencyGreen":
    """Given parameters of the auxiliary system, a single particle Green's
    function is constructed and its self-engergy/hybridization function
    returned

    [extended_summary]

    Parameters
    ----------
    auxsys : auxiliary_system_parameter.AuxiliarySystem
            Auxiliary system parameters class

    Returns
    -------
    out : FrequencyGreen
        self-energy of given Green's functions
    """
    green = FrequencyGreen(auxsys.ws)
    green.set_green_from_auxiliary(auxsys)
    return green.get_self_enerqy()


def keldysh_unity(freq: np.ndarray) -> FrequencyGreen:
    """Get the unity matrix regarding the keldysh contour, 1^R  and 0^K for
    given frequency grid.

    Parameters
    ----------
    freq : np.ndarray (dim,)
        1D Frequency grid

    Returns
    -------
    out: FrequencyGreen
        Green's function representing the unity matrix regarding the keldysh
        contour
    """
    return FrequencyGreen(freq=freq,
                          retarded=np.ones(freq.shape, dtype=np.complex128),
                          keldysh=np.zeros(freq.shape, dtype=np.complex128))


if __name__ == "__main__":

    # Setting up Auxiliary system parameters
    Nb = 1
    ws = np.linspace(-5, 5, 1001)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.1 + 0.0j, 0.0 + 0.0j, 0.2 + 0.0j])

    # initializing auxiliary system and E, Gamma1 and Gamma2 for a
    # particle-hole symmetric system
    aux = auxp.AuxiliarySystem(Nb, ws)
    aux.set_ph_symmetric_aux(es, ts, gamma)

    print("E: \n", np.array(aux.E))
    print("Gamma1: \n", aux.Gamma1)
    print("Gamma2: \n", aux.Gamma2)
    print("Gamma2-Gamma2: \n", (aux.Gamma2 - aux.Gamma1))

    green = FrequencyGreen(aux.ws)
    green.set_green_from_auxiliary(aux)

    plt.figure()
    plt.plot(aux.ws, green.retarded.imag)
    plt.plot(aux.ws, green.retarded.real)
    plt.plot(aux.ws, green.keldysh.imag)
    plt.plot(aux.ws, green.keldysh.real)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$ImG^R(\omega)$", r"$ReG^R(\omega)$",
               r"$ImG^K(\omega)$", r"$ReG^k(\omega)$"])
    plt.show()

    sigma = green.get_self_enerqy()

    plt.figure()
    plt.plot(aux.ws, sigma.retarded.imag)
    plt.plot(aux.ws, sigma.retarded.real)
    plt.plot(aux.ws, sigma.keldysh.imag)
    plt.plot(aux.ws, sigma.keldysh.real)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$Im\Delta^R_{aux}(\omega)$", r"$Re\Delta^R_{aux}(\omega)$",
               r"$Im\Delta^K_{aux}(\omega)$", r"$Re\Delta^K_{aux}(\omega)$"])
    plt.show()

    green2 = FrequencyGreen(aux.ws)
    green2.dyson(aux.ws, sigma)

    # checking that dyson equation from hybridization function results in
    # same Green's function
    plt.figure()
    plt.plot(aux.ws, np.abs(green2.retarded - green.retarded))
    plt.plot(aux.ws, np.abs(green2.keldysh - green.keldysh))
    plt.xlabel(r"$\omega$")
    plt.legend([r"$G^R_{aux}(\omega)$", r"$G^R_{aux}(\omega)$",
               r"$ImG^K_{aux}(\omega)$", r"$ReG^K_{aux}(\omega)$"])
    plt.show()
