# %%
from scipy.sparse import diags
import numpy as np
import matplotlib.pyplot as plt


class AuxiliarySystem:
    """The AuxiliarySystem class provides the matrices E, Gamma1, Gamma2
    necessary for calculating the Liouvillian [1](https://link.aps.org/doi/10.1103/PhysRevB.89.165105).
    The Hybridization function can be extracted from the class.

    Parameters
    ----------
        Nb : int
            Number of bath sites to the left/right
            (total number of bath sites is 2*Nb)

        ws : array_like
            1-D Frequency grid

    Attributes
    ----------
        Nb :        int
                    Number of bath sites

        N :         int
                    Number of sites (2*Nb+1)

        N_gamma :   int
                    Number of independent values in Gamma1 N_gamma = (N-1)N/2

        ws :        array_like
                    Frequency grid

        E:          numpy.ndarray (self.N,self.N)
                    T-Matrix (hopping and onsite potential)
                    of the Auxiliary Model

        Gamma1:     numpy.ndarray (self.N,self.N)
                    Coupling to Markovian bath 1

        Gamma2:     numpy.ndarray (self.N,self.N)
                    Coupling to Markovian bath 2
    """

    def __init__(self, Nb, ws) -> None:

        self.Nb = Nb
        self.N = 2 * self.Nb + 1
        self.N_gamma = int((self.N - 1) * self.N / 2)
        self.ws = ws

    def set_E_ph_symmetric(self, es, ts):
        """Sets particle-hole symmetric T-Matrix as class attribute E. E is of
        shape (self.N,self.N).

        Parameters
        ----------
        es : numpy.ndarray (self.Nb,)
            Onsite potentials of the reduced, auxiliary system

        ts : numpy.ndarray (self.Nb,)
            Hopping terms of the educed, auxiliary system
        """

        assert len(es) == self.Nb, "es doesn't have size Nb"
        assert len(ts) == self.Nb, "ts doesn't have size Nb"

        t = np.array([*ts, *(ts[::-1])], dtype=complex)
        E = np.array([*es, 0, *(es[::-1] * (-1))], dtype=complex)
        offset = [-1, 0, 1]
        self.E = diags([t, E, t], offset).toarray()

    def set_E_general(self, es, ts):
        """Sets general T-Matrix as class attribute E. E is of
        shape (self.N,self.N).

        Parameters
        ----------
        es : array_like
            Onsite potentials of the reduced, auxiliary system

        ts : array_like
            Hopping terms of the educed, auxiliary system
        """
        assert len(es) == self.N
        assert len(ts) == self.N - 1

        offset = [-1, 0, 1]
        self.E = diags([ts, es, ts], offset, dtype=complex).toarray()

    def get_Gamma_from_upper_tiagonal(self, G_upper):
        """Returns full gamma matrix (see [1]) from upper triangular matrix

        Parameters
        ----------
        G_upper : numpy.ndarray (dim,dim)
            Upper triangular matrix

        Returns
        -------
        out: numpy.ndarray (dim,dim)
            Lower triangular entries are obtained by adjoint of G_upper.
            Diagonal and upper triangular entries are equal to the diagonal of
            G_upper.
        """
        G_adj = np.copy(np.transpose(np.conj(G_upper)))
        G_adj[np.diag_indices(G_adj.shape[0])] = 0
        return G_upper + G_adj

    def get_Gamma_general(self, gammas):
        """Calculate full gamma matrix from array_like gamma containing all
        independent entries of gamma matrix.

        In order to preserve causality the entries at [f,i] and [i,f] are set
        to zero, where f is the impurity site located at the center and i runs
        from {0,...,self.N-1}. Therefore the Markovian baths don't couple to
        the impurity site.


        Parameters
        ----------
        gammas : array_like ((self.N-1)*self.N/2)
            Containing all independent entries of gamma matrix

        Returns
        -------
        out: numpy.ndarray (self.N,self.N)
            Gamma matrix describing the coupling between the auxiliary sites
            to a Markovian bath
        """
        assert self.N_gamma == len(
            gamma), ("gamma has wrong length, should be (N-1)N/2,"
                     + " with N = 2*self.Nb + 1")

        Gamma = np.zeros((self.N, self.N), dtype=complex)
        n = 0
        ff = self.Nb
        for i in range(self.N):
            if i != ff:
                for j in range(self.N):
                    if (j != ff) and (i <= j):
                        Gamma[i, j] = gammas[n]
                        n += 1
        return self.get_Gamma_from_upper_tiagonal(Gamma)

    def get_Gamma2_ph_symmetric(self, Gamma1):
        """Returns the Gamma2 matrix calculated from Gamma1 in the particle-hole
        symmetric case

        Parameters
        ----------
        Gamma1 : numpy.ndarray (dim,dim)
            Coupling of auxiliary sites to the Markovian bath 1

        Returns
        -------
        Gamma2: numpy.ndarray (dim,dim)
            Coupling of auxiliary sites to the Markovian bath 2
        """
        return np.array([[((-1)**(i + j)) * (Gamma1[((self.N - 1) - j), ((self.N - 1) - i)])
                          for i in range(self.N)] for j in range(self.N)])

    def set_ph_symmetric_aux(self, es, ts, gammas):
        """Set the matrices E, Gamma1,Gamma2 describing the auxiliary system in
        the particle-hole symmetric case.

        Parameters
        ----------
        es : numpy.ndarray (self.Nb,)
            Onsite potentials of the reduced, auxiliary system

        ts : numpy.ndarray (self.Nb,)
            Hopping terms of the educed, auxiliary system

        gammas : array_like ((self.N-1)*self.N/2)
            Containing all independent entries of gamma matrix
        """
        self.set_E_ph_symmetric(es, ts)
        self.Gamma1 = self.get_Gamma_general(gammas)
        self.Gamma2 = self.get_Gamma2_ph_symmetric(self.Gamma1)

    def set_general_aux(self, es, ts, gamma1, gamma2):
        """Set the matrices E, Gamma1,Gamma2 describing the auxiliary system in
        the general case.

        Parameters
        ----------
        es : array_like
            Onsite potentials of the reduced, auxiliary system

        ts : array_like
            Hopping terms of the educed, auxiliary system

        gamma1 : array_like ((self.N-1)*self.N/2)
            Containing all independent entries of gamma matrix Gamma1

        gamma2 : array_like ((self.N-1)*self.N/2)
            Containing all independent entries of gamma matrix Gamma2
        """
        self.set_E_general(es, ts)
        self.Gamma1 = self.get_Gamma_general(gamma1)
        self.Gamma2 = self.get_Gamma_general(gamma2)

    def get_green_aux(self):
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

        def z_aux(x):
            Z_R_aux = np.linalg.inv(x * np.identity(self.N) - self.E +
                                    1.0j * (self.Gamma2 + self.Gamma1))
            Z_A_aux = np.linalg.inv(x * np.identity(self.N) - self.E -
                                    1.0j * (self.Gamma2 + self.Gamma1))
            Z_K_aux = 2.0j * \
                (Z_R_aux.dot((self.Gamma2 - self.Gamma2).dot(Z_A_aux)))
            return (Z_R_aux[self.Nb, self.Nb], Z_K_aux[self.Nb, self.Nb])

        vec_z_aux = np.vectorize(z_aux)
        G_aux = vec_z_aux(self.ws)
        return G_aux[0], G_aux[1]  # G_R_aux,G_K_aux

    def get_auxiliary_hybridization(self, G_R_aux, G_K_aux):
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

        def get_hyb_aux(w, G_R, G_K):
            hyb_R = w - self.E[self.Nb, self.Nb] - 1.0 / G_R
            hyb_K = 1.0j * ((1 / (G_R * np.conj(G_R))) * G_K)
            return hyb_R, hyb_K

        vec_get_hyb_aux = np.vectorize(get_hyb_aux)
        hyb_aux = vec_get_hyb_aux(self.ws, G_R_aux, G_K_aux)
        return hyb_aux[0], hyb_aux[1]


# %%
if __name__ == "__main__":

    # Setting up Auxiliary system parameters
    Nb = 1
    ws = np.linspace(-5, 5, 1001)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.1 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])

    # initializing auxiliary system and E, Gamma1 and Gamma2 for a particle-hole
    # symmetric system
    aux = AuxiliarySystem(Nb, ws)
    aux.set_ph_symmetric_aux(es, ts, gamma)

    print("E: \n", np.array(aux.E))
    print("Gamma1: \n", aux.Gamma1)
    print("Gamma2: \n", aux.Gamma2)
    print("Gamma2-Gamma2: \n", (aux.Gamma2 - aux.Gamma1))

    # G_aux = aux.get_green_aux()
    G_R_aux, G_K_aux = aux.get_green_aux()

    plt.figure()
    plt.plot(aux.ws, G_R_aux.imag)
    plt.plot(aux.ws, G_R_aux.real)
    plt.xlabel(r"$\omega$")
    plt.legend(["$ImG^R(\omega)$", "$ReG^R(\omega)$"])
    plt.show()

    hyb_R, hyb_K = aux.get_auxiliary_hybridization(G_R_aux, G_K_aux)

    plt.figure()
    plt.plot(aux.ws, hyb_R.imag)
    plt.plot(aux.ws, hyb_R.real)
    plt.xlabel(r"$\omega$")
    plt.legend(["$Im\Delta^R_{aux}(\omega)$", "$Re\Delta^R_{aux}(\omega)$"])
    plt.show()

# TODO: The rest should be seperated
#      2. Given matrices and frequencies ws the local Green's should be
#         calculated.
#      3. From the Green's function the hybridization should be returned.
#      4. A costfunction should be written.
#      5. An optimization routine has to be written.
#
# TODO: in the non-particle-hole symmetric case Gamma1 and Gamma2 are independent
# TODO: Later Extend this to multiorbital case

# %%
