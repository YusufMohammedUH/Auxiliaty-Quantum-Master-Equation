# %%
import numpy as np
import src.auxiliary_system_parameter as auxp
import matplotlib.pyplot as plt


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

    def __init__(self, freq: np.ndarray, retarded=None, keldysh=None) -> None:
        if not isinstance(freq, np.ndarray):
            raise TypeError("freq must be of type numpy.array!")
        if (not isinstance(retarded, np.ndarray)) and (retarded is not None):
            raise TypeError("retarded must be of type numpy.array or None!")
        if (not isinstance(keldysh, np.ndarray)) and (keldysh is not None):
            raise TypeError("keldysh must be of type numpy.array or None!")

        self.freq = freq
        if retarded is None:
            self.retarded = np.zeros(len(freq), dtype=complex)
        else:
            if freq.shape != retarded.shape:
                raise ValueError("freq and retarded must have same shape")
            self.retarded = retarded
        if keldysh is None:
            self.keldysh = np.zeros(len(freq), dtype=complex)
        else:
            if freq.shape != keldysh.shape:
                raise ValueError("freq and keldysh must have same shape")
            self.keldysh = keldysh

    def __add__(a: "FrequencyGreen", b: "FrequencyGreen") -> "FrequencyGreen":
        """Add two FrequencyGreen objects

        Parameters
        ----------
        a : FrequencyGreen

        b : FrequencyGreen

        Returns
        -------
        out: FrequencyGreen
        """
        return FrequencyGreen(a.freq, a.retarded + b.retarded,
                              a.keldysh + b.keldysh)

    def __mul__(a: "FrequencyGreen", b: "FrequencyGreen") -> "FrequencyGreen":
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
        a : FrequencyGreen

        b : FrequencyGreen

        Returns
        -------
        out: FrequencyGreen
        """
        return FrequencyGreen(a.freq, a.retarded * b.retarded,
                              a.retarded * b.keldysh +
                              a.keldysh * b.retarded.conj())

    def dyson(self, green_0_ret_inverse, self_energy) -> None:
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

        def z_aux(x):
            Z_R_aux = np.linalg.inv(x * np.identity(auxsys.N) - auxsys.E +
                                    1.0j * (auxsys.Gamma2 + auxsys.Gamma1))
            Z_A_aux = np.linalg.inv(x * np.identity(auxsys.N) - auxsys.E -
                                    1.0j * (auxsys.Gamma2 + auxsys.Gamma1))
            Z_K_aux = 2.0j * \
                (Z_R_aux.dot((auxsys.Gamma2 - auxsys.Gamma1).dot(Z_A_aux)))
            return (Z_R_aux[auxsys.Nb, auxsys.Nb],
                    Z_K_aux[auxsys.Nb, auxsys.Nb])

        vec_z_aux = np.vectorize(z_aux)
        G_aux = vec_z_aux(self.freq)

        self.retarded = G_aux[0]
        self.keldysh = G_aux[1]

    def get_self_enerqy(self, green_0_ret_inverse=None) -> "FrequencyGreen":
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

        def get_self_enerqy(green_0_ret_inverse_w, green_ret_w, green_kel_w):
            green_ret_w_inverse = 0
            if green_ret_w == 0:
                green_ret_w_inverse = float('inf')
            else:
                green_ret_w_inverse = 1.0 / green_ret_w
            sigma_ret_w = green_0_ret_inverse_w - green_ret_w_inverse
            green_ret_w_abs_square_inverse = 0
            if (green_ret_w * np.conj(green_ret_w)) == 0:
                green_ret_w_abs_square_inverse = float('inf')
            else:
                green_ret_w_abs_square_inverse = (
                    1 / (green_ret_w * np.conj(green_ret_w)))
            sigma_kel_w = (
                green_ret_w_abs_square_inverse * green_kel_w)
            return sigma_ret_w, sigma_kel_w

        vec_get_self_enerqy = np.vectorize(get_self_enerqy)
        sigma = vec_get_self_enerqy(
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


# %%
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

# %%
