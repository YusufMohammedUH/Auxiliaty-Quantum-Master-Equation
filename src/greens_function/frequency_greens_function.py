# %%
from typing import Tuple, Union
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import src.auxiliary_mapping.auxiliary_system_parameter as auxp
import src.util.hdf5_util as hd5
import src.greens_function.keldysh_identity as kid


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
    """Simple frequency Green's function container. Contains the
    single particle Green's on the Keldysh contour in Keldysh rotated
    representation, e.g. G^R and G^K. G^A is dropt do to symmetry [G^R]* = G^A.

    Parameters
    ----------
    freq : numpy.ndarry (dim,)
        1-D frequency grid

    retarded : numpy.ndarry (dim,), optional
        Contains the retarded Green's

    keldysh : numpy.ndarry (dim,), optional
        Contains the keldysh/lesser/greater Green's

    Attributes
    ----------
    freq : numpy.ndarry (dim,)
        1-D Frequency grid

    retarded : numpy.ndarry (dim,)
        Contains the retarded Green's

    keldysh : numpy.ndarry (dim,)
        Contains the keldysh Green's


    Raises
    ------
    TypeError
        "freq must be of type numpy.array!"

    TypeError
        "retarded must be of type numpy.array or None!"

    TypeError
        "keldysh must be of type numpy.array or None!"

    ValueError
        "freq and retarded must have same shape"

    ValueError
        "freq and keldysh must have same shape"
    """

    def __init__(self, freq: np.ndarray,
                 retarded: Union[np.ndarray, None] = None,
                 keldysh: Union[np.ndarray, None] = None,
                 fermionic: bool = True,
                 keldysh_comp: str = 'lesser', orbitals=1) -> None:
        """Initialize self.  See help(type(self)) for accurate signature.
        """
        if not isinstance(freq, np.ndarray):
            raise TypeError("ERROR: freq must be of type numpy.array!")
        if (not isinstance(retarded, np.ndarray)) and (retarded is not None):
            raise TypeError(
                "ERROR: retarded must be of type numpy.array or None!")
        if (not isinstance(keldysh, np.ndarray)) and (keldysh is not None):
            raise TypeError(
                "ERROR: keldysh must be of type numpy.array or None!")
        if keldysh_comp != 'lesser' and keldysh_comp != 'keldysh':
            raise ValueError("Error: keldysh_comp has to be either 'lesse'" +
                             " of 'keldysh'")
        self.orbitals = orbitals
        self.keldysh_comp = keldysh_comp
        self.fermionic = fermionic
        self.freq = freq
        if self.freq.flags.writeable:
            self.freq.flags.writeable = False
        if retarded is None:
            self.retarded = np.zeros(len(freq), dtype=np.complex128)
        else:
            if freq.shape != retarded.shape:
                raise ValueError(
                    "ERROR: freq and retarded must have same shape")
            self.retarded = np.copy(retarded)
        if keldysh is None:
            self.keldysh = np.zeros(len(freq), dtype=np.complex128)
        else:
            if freq.shape != keldysh.shape:
                raise ValueError(
                    "ERROR: freq and keldysh must have same shape")
            self.keldysh = np.copy(keldysh)

    def copy(self) -> "FrequencyGreen":
        """Return a copy of the object.

        Returns
        -------
        out: FrequencyGreen
        """
        return FrequencyGreen(self.freq, self.retarded.copy(),
                              self.keldysh.copy(),
                              fermionic=self.fermionic,
                              orbitals=self.orbitals,
                              keldysh_comp=self.keldysh_comp)

    def get_changed_keldysh_comp(self, keldysh_comp: str) -> "FrequencyGreen":
        """Return a copy of the object with changed keldysh_comp.

        Parameters
        ----------
        keldysh_comp : str
            New keldysh_comp

        Returns
        -------
        out: FrequencyGreen
        """
        if self.keldysh_comp == keldysh_comp:
            return self.copy()
        elif keldysh_comp == 'lesser':

            return FrequencyGreen(freq=self.freq,
                                  retarded=self.retarded.copy(),
                                  keldysh=self.get_lesser().copy(),
                                  fermionic=self.fermionic,
                                  orbitals=self.orbitals,
                                  keldysh_comp=keldysh_comp)
        elif keldysh_comp == 'keldysh':
            return FrequencyGreen(freq=self.freq,
                                  retarded=self.retarded.copy(),
                                  keldysh=self.get_keldysh().copy(),
                                  fermionic=self.fermionic,
                                  keldysh_comp=keldysh_comp,
                                  orbitals=self.orbitals)
        else:
            raise ValueError("Error: keldysh_comp has to be either 'lesser'" +
                             " or 'keldysh'")

    def __add__(self, other: Union["FrequencyGreen", "kid.KeldyshIdentity",
                                   int, float, complex]) -> "FrequencyGreen":
        """Add FrequencyGreen or KeldyshIdentity objects or number from
        current object and return resulting FrequencyGreen.

        In case of a number 'other' and Green's function 'G' and the identity
        matrix 'I' in the Keldysh contour

        other*I+G

        is returned

        Parameters
        ----------
        other : FrequencyGreen

        Returns
        -------
        out: FrequencyGreen
        """
        if isinstance(other, FrequencyGreen):
            if self.keldysh_comp != other.keldysh_comp:
                raise ValueError("Error: keldysh_comp has to be the same " +
                                 "for both objects")
            if self.fermionic != other.fermionic:
                raise ValueError("Error: fermionic has to be the same for " +
                                 "both objects")
            if self.orbitals != other.orbitals:
                raise ValueError("Error: orbitals has to be the same for " +
                                 "both objects")
            if self.orbitals == 1:
                return FrequencyGreen(freq=self.freq,
                                      retarded=self.retarded + other.retarded,
                                      keldysh=self.keldysh + other.keldysh,
                                      fermionic=self.fermionic,
                                      keldysh_comp=self.keldysh_comp,
                                      orbitals=self.orbitals)
            else:
                raise ValueError("Error: addition of Green's functions" +
                                 " with orbitals > 1 not implemented yet")
        elif (isinstance(other, int) or isinstance(other, float)
              or isinstance(other, complex)):
            return FrequencyGreen(freq=self.freq,
                                  retarded=self.retarded + other,
                                  keldysh=self.keldysh,
                                  fermionic=self.fermionic,
                                  keldysh_comp=self.keldysh_comp,
                                  orbitals=self.orbitals)
        elif isinstance(other, kid.KeldyshIdentity):
            return other + self

    def __sub__(self, other: Union["FrequencyGreen", "kid.KeldyshIdentity",
                                   int, float, complex]) -> "FrequencyGreen":
        """Subtract FrequencyGreen or KeldyshIdentity objects or number from
        current object and return resulting FrequencyGreen.

        In case of a number 'other' and Green's function 'G' and the identity
        matrix 'I' in the Keldysh contour

        other*I-G

        is returned

        Parameters
        ----------
        other : FrequencyGreen

        Returns
        -------
        out: FrequencyGreen
        """
        if isinstance(other, FrequencyGreen):
            if self.keldysh_comp != other.keldysh_comp:
                raise ValueError("Error: keldysh_comp has to be the same " +
                                 "for both objects")
            if self.fermionic != other.fermionic:
                raise ValueError("Error: fermionic has to be the same for " +
                                 "both objects")
            if self.orbitals != other.orbitals:
                raise ValueError("Error: orbitals has to be the same for " +
                                 "both objects")
            if self.orbitals == 1:
                return FrequencyGreen(freq=self.freq,
                                      retarded=self.retarded - other.retarded,
                                      keldysh=self.keldysh - other.keldysh,
                                      fermionic=self.fermionic,
                                      keldysh_comp=self.keldysh_comp,
                                      orbitals=self.orbitals)
            else:
                raise ValueError("Error: subtraction of Green's functions" +
                                 " with orbitals > 1 not implemented yet")
        elif (isinstance(other, int) or isinstance(other, float)
              or isinstance(other, complex)):
            return FrequencyGreen(self.freq, self.retarded - other,
                                  self.keldysh,
                                  fermionic=self.fermionic,
                                  keldysh_comp=self.keldysh_comp,
                                  orbitals=self.orbitals)
        elif isinstance(other, kid.KeldyshIdentity):
            return (other - self) * (-1)

    def __mul__(self, other: Union["FrequencyGreen", "kid.KeldyshIdentity",
                                   int, float, complex]) -> "FrequencyGreen":
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
            if self.keldysh_comp != other.keldysh_comp:
                raise ValueError("Error: keldysh_comp has to be the same " +
                                 "for both objects")
            if self.orbitals != other.orbitals:
                raise ValueError("Error: orbitals has to be the same for " +
                                 "both objects")
            if self.orbitals == 1:
                return FrequencyGreen(freq=self.freq,
                                      retarded=(
                                          self.retarded * other.retarded),
                                      keldysh=(
                                          self.retarded * other.keldysh +
                                          self.keldysh * other.retarded.conj()
                                      ),
                                      fermionic=self.fermionic,
                                      keldysh_comp=self.keldysh_comp,
                                      orbitals=self.orbitals)
            else:
                raise ValueError("Error: multiplication of Green's functions" +
                                 " with orbitals > 1 not implemented yet")
        elif (isinstance(other, int) or isinstance(other, float)
              or isinstance(other, complex)):
            return FrequencyGreen(freq=self.freq,
                                  retarded=self.retarded * other,
                                  keldysh=self.keldysh * other,
                                  fermionic=self.fermionic,
                                  keldysh_comp=self.keldysh_comp,
                                  orbitals=self.orbitals)
        elif isinstance(other, kid.KeldyshIdentity):
            return self

    def get_spectral_func(self) -> np.ndarray:
        """Return the spectral function.
        """
        if self.fermionic:
            tmp = (-1 / (2 * np.pi)) * \
                (self.retarded - self.retarded.conj()).imag
            return tmp
        elif not self.fermionic:
            tmp = (-1 / (2 * np.pi)) * \
                (self.retarded - self.get_advanced()).imag
            return tmp
        raise AttributeError("ERROR: self.fermionic is not boolian")

    def get_lesser(self) -> np.ndarray:
        """Return the lesser component
        """
        if self.keldysh_comp == "lesser":
            return self.keldysh

        elif self.keldysh_comp == "keldysh":
            return 0.5 * (self.keldysh + 1j * self.get_spectral_func())

    def get_greater(self) -> np.ndarray:
        """Return greater component
        """
        return self.get_keldysh() - self.get_lesser()

    def get_time_ordered(self) -> np.ndarray:
        """ Return the time ordered Green's function
        """
        return 0.5 * (self.retarded + self.get_advanced() + self.get_keldysh())

    def get_anti_time_ordered(self) -> np.ndarray:
        """ Return the anti-time ordered Green's function
        """
        return 0.5 * (self.get_keldysh() - (self.retarded + self.get_advanced()
                                            ))

    def get_keldysh(self) -> np.ndarray:
        """Return the keldysh component
        """
        if self.keldysh_comp == "keldysh":
            return self.keldysh
        elif self.keldysh_comp == "lesser":
            return -1j * self.get_spectral_func() + 2. * self.keldysh

    def get_advanced(self) -> np.ndarray:
        """Return the advanced component.

        Returns
        -------
        out: np.ndarray
            advanced component
        """
        if self.fermionic:
            if self.orbitals > 1:
                np.array(list(map(lambda x: x.H, self.retarded)))
            else:
                return self.retarded.conj()
        else:
            # XXX: Advanced for multi orbital systems have to be included
            return self.retarded[::-1]

    def inverse(self) -> "FrequencyGreen":
        """Return the inverse Green's function

        Returns
        -------
        out: FrequencyGreen
            Inverse Green's function
        """
        if self.orbitals > 1:
            retarded_inv = np.array(list(map(np.linalg.inv, self.retarded)))
            advanced_inv = np.array(list(map(lambda x: x.H, retarded_inv)))
        else:
            retarded_inv = np.array([1. / ret if np.abs(ret) != np.inf
                                     else 0 for ret in self.retarded],
                                    dtype=np.complex128)
            advanced_inv = retarded_inv.conj()

        keldysh_inv = np.array([-1. * ret * kel * adv if np.abs(ret) != 0
                                else 0 for ret, kel, adv in zip(
            retarded_inv, self.keldysh, advanced_inv)],
            dtype=np.complex128)
        return FrequencyGreen(freq=self.freq,
                              retarded=retarded_inv,
                              keldysh=keldysh_inv,
                              fermionic=self.fermionic,
                              keldysh_comp=self.keldysh_comp,
                              orbitals=self.orbitals)

    def get_inverse_no_keldysh_rot(self, component: Tuple):
        """return the inverse green's function component of
        green's function without the keldysh rotation, e.g.

        | G^{T} G^{<}      |
        | G^{>} G^{\bar{T}}|

        using
                                | d   -b |
        A^{-1} = \frac{1}{ad-bc}| -c   a |

        Parameters
        ----------
        component : Tuple
            (00): time-ordered ,(01): lesser, (10): greater,
            (11): anti-time-ordered
        """
        assert component in [(0, 0), (0, 1), (1, 0), (1, 1)]
        # XXX: We assume the
        if self.orbitals > 1:
            tmps = np.array([time_ord.dot(anti_time) - les.dot(great)
                            for time_ord, les, great, anti_time in
                            zip(self.get_time_ordered(),
                                self.get_lesser(),
                                self.get_greater(),
                                self.get_anti_time_ordered())],
                            dtype=np.complex128)
            tmps = np.array(list(map(np.linalg.inv, tmps)))
            if component == (0, 0):
                green_inv = np.array([tmp.dot(anti_time) for anti_time, tmp in
                                      zip(self.get_anti_time_ordered(), tmps)],
                                     dtype=np.complex128)
            elif component == (0, 1):
                green_inv = np.array([-1. * tmp.dot(les) for les, tmp in
                                      zip(self.get_lesser(), tmps)],
                                     dtype=np.complex128)
            elif component == (1, 0):
                green_inv = np.array([-1. * tmp.dot(great) for great, tmp in
                                      zip(self.get_greater(), tmps)],
                                     dtype=np.complex128)
            elif component == (1, 1):
                green_inv = np.array([tmp.dot(time_ord) for time_ord, tmp in
                                      zip(self.get_time_ordered(), tmps)],
                                     dtype=np.complex128)
        else:
            tmps = (self.get_time_ordered() * self.get_anti_time_ordered()
                    - self.get_lesser() * self.get_greater())
            tmps = np.array([1. / tmp if np.abs(tmp) != np.inf
                            else 0 for tmp in tmps],
                            dtype=np.complex128)
            if component == (0, 0):
                green_inv = tmps * self.get_anti_time_ordered()
            elif component == (0, 1):
                green_inv = -1. * tmps * self.get_lesser()
            elif component == (1, 0):
                green_inv = -1. * tmps * self.get_greater()
            elif component == (1, 1):
                green_inv = tmps * self.get_time_ordered()
            return green_inv

    def dyson(self, self_energy: "FrequencyGreen", e_tot: float = 0,
              g0_inv: Union["FrequencyGreen", np.ndarray, None] = None,
              g0: Union["FrequencyGreen", np.ndarray, None] = None) -> None:
        """Calculate and set the the frequency Green's function, through the
        Dyson equation for given self-energy sigma.

        Parameters
        ----------
        self_energy : FrequencyGreen
            Self-energy could be only the hybridization/embeding self-energy
            and/or include the interacting self-energy

        e_tot : float, optional
            Onsite energie, by default 0

        g0_inv: np.array, optional
            Inverse non-interacting Green's function, by default None
        """
        if self.keldysh_comp != self_energy.keldysh_comp:
            raise ValueError("Keldysh components of Green's function and "
                             "self-energy are not equal.")

        if self.fermionic != self_energy.fermionic:
            raise ValueError("Fermionicity of Green's function and "
                             "self-energy are not equal.")
        if self.orbitals != self_energy.orbitals:
            raise ValueError("Number of orbitals of Green's function and "
                             "self-energy are not equal.")

        if (g0 is not None) and (g0_inv is None):
            if isinstance(g0, FrequencyGreen):
                if self.orbitals != g0.orbitals:
                    raise ValueError("Orbitals of Green's function and"
                                     " non-interacting Green's function"
                                     " are not equal.")
                if self.keldysh_comp != g0.keldysh_comp:
                    raise ValueError("Keldysh components of Green's function"
                                     " and non-interacting Green's function"
                                     " are not equal.")
                if self.fermionic != g0.fermionic:
                    raise ValueError("Fermionicity of Green's function and"
                                     " non-interacting Green's function"
                                     " are not equal.")
                g0_mul_sigma_ret_tmp = g0.retarded * self_energy.retarded
                g0_ret_tmp = g0.retarded
            elif isinstance(g0, np.ndarray):
                # TODO: waring should be implemented
                #       it can't be checked if array is fermionic or
                #       bosonic

                # g0 is now really g0.retarded
                if len(g0[0]) != self.orbitals:
                    raise ValueError("Orbitals of Green's function and"
                                     " g0 are not equal.")
                g0_mul_sigma_ret_tmp = g0 * self_energy.retarded
                g0_ret_tmp = g0
            else:
                raise TypeError("g0 has to by of type numpy.nparray,"
                                " FrequencyGreen or None")

            if self.orbitals > 1:
                retarded = np.array(
                    list(map(np.linalg.inv, (
                        1. - g0_mul_sigma_ret_tmp))
                    ))
            else:
                retarded = np.array([1. / ret if np.abs(ret) != np.inf
                                     else 0 for ret
                                     in (
                    1. - g0_mul_sigma_ret_tmp)
                ], dtype=np.complex128)
            self.retarded = retarded * g0_ret_tmp
        elif (g0_inv is not None) and (g0 is None):
            if isinstance(g0_inv, np.ndarray):
                if len(g0_inv[0]) != self.orbitals:
                    raise ValueError("Orbitals of Green's function and"
                                     " g0_inv are not equal.")
                g0_inv_min_sigma_ret_tmp = g0_inv - self_energy.retarded
            elif isinstance(g0_inv, FrequencyGreen):
                if self.orbitals != g0_inv.orbitals:
                    raise ValueError("Orbitals of Green's function and"
                                     " inverse non-interacting Green's"
                                     " function are not equal.")
                if self.keldysh_comp != g0_inv.keldysh_comp:
                    raise ValueError("Keldysh components of Green's"
                                     " function and inverse"
                                     " non-interacting Green's function"
                                     " are not equal.")
                if self.fermionic != g0_inv.fermionic:
                    raise ValueError("Fermionicity of Green's function and"
                                     " inverse non-interacting Green's"
                                     " function are not equal.")
                g0_inv_min_sigma_ret_tmp = g0_inv.retarded \
                    - self_energy.retarded
            else:
                raise TypeError("g0_inv has to by of type numpy.nparray,"
                                " FrequencyGreen or None")
            if self.orbitals > 1:
                retarded = np.array(
                    list(map(np.linalg.inv, (
                        g0_inv_min_sigma_ret_tmp))
                    ))
            else:
                retarded = np.array([
                    1. / ret if np.abs(ret) != np.inf else 0 for ret
                    in (g0_inv_min_sigma_ret_tmp)
                ], dtype=np.complex128)

            self.retarded = retarded
        elif (g0 is None) and (g0_inv is None):
            if self.orbitals > 1:
                identity_mat = np.eye(self.orbitals)
                g0_inv = np.array(
                    [w * identity_mat - e_tot for w in self.freq])
                self.retarded = np.array(
                    list(map(np.linalg.inv, (
                        g0_inv - self_energy.retarded))
                    ))
            else:
                g0_inv = self.freq - e_tot
                self.retarded = np.array([
                    1. / ret if np.abs(ret) != np.inf else 0 for ret
                    in (self.freq - e_tot - self_energy.retarded)],
                    dtype=np.complex128)
        else:
            raise TypeError("input of g0 and go_inv are exclusive.")
        self.keldysh = (self.retarded * self_energy.keldysh *
                        self.get_advanced())

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
        if self.keldysh_comp == 'lesser':
            self.keldysh_comp = 'keldysh'
            self.keldysh = self.get_lesser()
            self.keldysh_comp = 'lesser'

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

        return FrequencyGreen(freq=self.freq,
                              retarded=sigma[0],
                              keldysh=sigma[1],
                              fermionic=self.fermionic,
                              keldysh_comp=self.keldysh_comp,
                              orbitals=self.orbitals)

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
                          {"freq_min": self.freq[0], "freq_max": self.freq[-1],
                           'N_freq': len(self.freq)})

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
            if (attrs['freq_max'] != self.freq[-1] or
                attrs['freq_min'] != self.freq[0]
                    or attrs['N_freq'] != self.freq.shape[0]):
                raise ValueError("Frequency grid of loaded data doesn't" +
                                 " match object frequency grid.")

        self.retarded = hd5.read_data(file=fname,
                                      dir_=f"{dir_}/{dataname}",
                                      dataname='retarded')
        self.keldysh = hd5.read_data(file=fname, dir_=f"{dir_}/{dataname}",
                                     dataname='keldysh')


def get_hyb_from_aux(auxsys: auxp.AuxiliarySystem, keldysh_comp: str
                     ) -> "FrequencyGreen":
    """Given parameters of the auxiliary system, a single particle Green's
    function is constructed and its self-engergy/hybridization function
    returned

    Parameters
    ----------
    auxsys : auxiliary_system_parameter.AuxiliarySystem
            Auxiliary system parameters class

    keldysh_comp: str, 'lesser' or 'keldysh'
                Specify which keldysh component is desired
    Returns
    -------
    out : FrequencyGreen
        self-energy of given Green's functions
    """
    green = FrequencyGreen(freq=auxsys.ws, keldysh_comp=keldysh_comp)
    green.set_green_from_auxiliary(auxsys)
    return green.get_self_enerqy()


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
    green2.dyson(self_energy=sigma)

    # checking that dyson equation from hybridization function results in
    # same Green's function
    plt.figure()
    plt.plot(aux.ws, np.abs(green2.retarded - green.retarded))
    plt.plot(aux.ws, np.abs(green2.keldysh - green.keldysh))
    plt.xlabel(r"$\omega$")
    plt.legend([r"$G^R_{aux}(\omega)$", r"$G^R_{aux}(\omega)$",
               r"$ImG^K_{aux}(\omega)$", r"$ReG^K_{aux}(\omega)$"])
    plt.show()

# %%
