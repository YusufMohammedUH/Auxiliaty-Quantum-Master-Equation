from scipy.sparse import diags
import numpy as np


class AuxiliarySystem:
    """The AuxiliarySystem class provides the matrices E, Gamma1, Gamma2
    necessary for calculating the Liouvillian
    [1](https://link.aps.org/doi/10.1103/PhysRevB.89.165105).
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

    def __init__(self, Nb: int, ws: np.ndarray) -> None:

        self.Nb = Nb
        self.N = 2 * self.Nb + 1
        self.N_gamma = int((self.N - 1) * self.N / 2)
        self.ws = ws

    def set_E_ph_symmetric(self, es: np.ndarray, ts: np.ndarray) -> None:
        """Sets particle-hole symmetric T-Matrix as class attribute E. E is of
        shape (self.N,self.N).

        Parameters
        ----------
        es : numpy.ndarray (self.Nb,)
            Onsite potentials of the reduced, auxiliary system

        ts : numpy.ndarray (self.Nb,)
            Hopping terms of the reduced, auxiliary system
        """

        assert len(es) == self.Nb, "es doesn't have size Nb"
        assert len(ts) == self.Nb, "ts doesn't have size Nb"

        t = np.array([*ts, *(ts[::-1])], dtype=np.complex64)
        E = np.array([*es, 0, *(es[::-1] * (-1))], dtype=np.complex64)
        offset = [-1, 0, 1]
        self.E = diags([t, E, t], offset).toarray()

    def set_E_general(self, es: np.ndarray, ts: np.ndarray) -> None:
        """Sets general T-Matrix as class attribute E. E is of
        shape (self.N,self.N).

        Parameters
        ----------
        es : array_like
            Onsite potentials of the reduced, auxiliary system

        ts : array_like
            Hopping terms of the educed, auxiliary system
        """
        assert len(es) == self.N, "es doesn't have size N"
        assert len(ts) == self.N - 1, "ts doesn't have size N-1"

        offset = [-1, 0, 1]
        self.E = diags([ts, es, ts], offset, dtype=np.complex64).toarray()

    def get_Gamma_from_upper_tiagonal(self, G_upper: np.ndarray) -> np.ndarray:
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
        G_adj = np.copy(G_upper.conj().T)
        G_adj[np.diag_indices(G_adj.shape[0])] = 0
        return G_upper + G_adj

    def get_Gamma_general(self, gammas: np.ndarray) -> np.ndarray:
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
            gammas), ("gamma has wrong length, should be (N-1)N/2,"
                      + " with N = 2*self.Nb + 1")

        Gamma = np.zeros((self.N, self.N), dtype=np.complex64)
        n = 0
        ff = self.Nb
        for i in range(self.N):
            if i != ff:
                for j in range(self.N):
                    if (j != ff) and (i <= j):
                        Gamma[i, j] = gammas[n]
                        n += 1
        return self.get_Gamma_from_upper_tiagonal(Gamma)

    def get_Gamma2_ph_symmetric(self, Gamma1: np.ndarray) -> np.ndarray:
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
        return np.array([[((-1)**(i + j)) * (Gamma1[((self.N - 1) - j),
                                                    ((self.N - 1) - i)])
                          for i in range(self.N)] for j in range(self.N)])

    def set_ph_symmetric_aux(self, es: np.ndarray, ts: np.ndarray,
                             gammas: np.ndarray) -> None:
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

    def set_general_aux(self, es: np.ndarray, ts: np.ndarray,
                        gamma1: np.ndarray, gamma2: np.ndarray) -> None:
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


if __name__ == "__main__":

    # Setting up Auxiliary system parameters
    Nb = 1
    ws = np.linspace(-5, 5, 1001)
    es = np.array([1])
    ts = np.array([0.5])
    gamma = np.array([0.1 + 0.0j, 0.0 + 0.0j, 0.1 + 0.0j])

    # initializing auxiliary system and E, Gamma1 and Gamma2 for a
    # particle-hole symmetric system
    aux = AuxiliarySystem(Nb, ws)
    aux.set_ph_symmetric_aux(es, ts, gamma)

    print("E: \n", np.array(aux.E))
    print("Gamma1: \n", aux.Gamma1)
    print("Gamma2: \n", aux.Gamma2)
    print("Gamma2-Gamma2: \n", (aux.Gamma2 - aux.Gamma1))


# TODO: Should be able to set target impurity site at beginning of in the
#       middel of the auxiliary problem, for now the impurity site is
#       positioned in the middel by default.
# TODO: In the non-particle-hole symmetric case Gamma1 and Gamma2 are
#       Independent
# TODO: Later Extend this to multiorbital case
