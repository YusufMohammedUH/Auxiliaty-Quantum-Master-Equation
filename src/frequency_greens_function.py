# %%
import numpy as np
import auxiliary_system_parameter as auxp
import matplotlib.pyplot as plt


class FrequencyGreen:
    def __init__(self, freq, retarded=None, keldysh=None) -> None:
        assert freq.shape[0] == len(freq)
        self.freq = freq
        if retarded is None:
            self.retarded = np.zeros(len(freq), dtype=complex)
        else:
            self.retarded = retarded
        if keldysh is None:
            self.keldysh = np.zeros(len(freq), dtype=complex)
        else:
            self.keldysh = keldysh

    def __add__(a, b):
        return FrequencyGreen(a.freq, a.retarded + b.retarded,
                              a.keldysh + b.keldysh)

    def set_green_from_auxiliary(self, auxsys):
        """Return the retarded and Keldysh Green's function of the impurity
        site located at self.Nb from self.E, self.Gamma1, self.Gamma2

        Returns
        -------
        G_R_aux : numpy.ndarray (self.N,)
                    Retarded Green's function of the impurity site in frequency
                    domane

        G_K_aux : numpy.ndarray (self.Nb,)
                    Keldysh Green's function of the impurity site in frequency
                    domane

        """
        assert np.array_equal(auxsys.ws, self.freq)

        def z_aux(x):
            Z_R_aux = np.linalg.inv(x * np.identity(auxsys.N) - auxsys.E +
                                    1.0j * (auxsys.Gamma2 + auxsys.Gamma1))
            Z_A_aux = np.linalg.inv(x * np.identity(auxsys.N) - auxsys.E -
                                    1.0j * (auxsys.Gamma2 + auxsys.Gamma1))
            Z_K_aux = 2.0j * \
                (Z_R_aux.dot((auxsys.Gamma2 - auxsys.Gamma2).dot(Z_A_aux)))
            return (Z_R_aux[auxsys.Nb, auxsys.Nb],
                    Z_K_aux[auxsys.Nb, auxsys.Nb])

        vec_z_aux = np.vectorize(z_aux)
        G_aux = vec_z_aux(self.freq)
        self.retarded = G_aux[0]
        self.keldysh = G_aux[1]

    def get_self_enerqy(self, green_0_ret_inverse=None):
        """Returns the retarded and Keldysh component of the self-energy
        from the retarded and Keldysh Green's.

        In case of no interacting self-energy the resulting self-energy is
        the embeding self-energy or hybridization function.

        Parameters
        ----------
        G_R_aux : numpy.ndarray (self.ws,)
            Green's function

        G_K_aux : numpy.ndarray (self.ws,)
            Green's function

        Returns
        -------
        hyb_R_aux : numpy.ndarray (self.N,)
                    Retarded self-energy of given Green's functions

        hyb_K_aux : numpy.ndarray (self.Nb,)
                    Keldysh self-energy of given Green's functions
        """
        if green_0_ret_inverse is None:
            green_0_ret_inverse = self.freq

        def get_self_enerqy(green_0_ret_inverse_w, green_ret_w, green_kel_w):
            sigma_ret_w = green_0_ret_inverse_w - 1.0 / green_ret_w
            sigma_kel_w = 1.0j * \
                ((1 / (green_ret_w * np.conj(green_ret_w))) * green_kel_w)
            return sigma_ret_w, sigma_kel_w

        vec_get_self_enerqy = np.vectorize(get_self_enerqy)
        sigma = vec_get_self_enerqy(
            green_0_ret_inverse, self.retarded, self.keldysh)

        return FrequencyGreen(self.freq, sigma[0], sigma[1])


# %%
if __name__ == "__main__":

    # Setting up Auxiliary system parameters
    Nb = 1
    ws = np.linspace(-5, 5, 1001)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.1 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])

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
    plt.xlabel(r"$\omega$")
    plt.legend([r"$ImG^R(\omega)$", r"$ReG^R(\omega)$"])
    plt.show()

    sigma = green.get_self_enerqy()

    plt.figure()
    plt.plot(aux.ws, sigma.retarded.imag)
    plt.plot(aux.ws, sigma.retarded.real)
    plt.xlabel(r"$\omega$")
    plt.legend([r"$Im\Delta^R_{aux}(\omega)$", r"$Re\Delta^R_{aux}(\omega)$"])
    plt.show()

# %%
