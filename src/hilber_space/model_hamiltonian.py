from typing import Tuple, Union
import numpy as np
from scipy import sparse
import src.hilber_space.define_fock_space_operators as op


def hubbard_U_center(nsite: int, U: float) -> np.ndarray:
    """Return an on-site interaction matrix with dimension (nsite,).
    The central element at (nsite-1)/2 equals U all other elements vanish.


    Parameters
    ----------
    nsite : int
        number of sites/orbitals in the system. Has to be odd.

    U : float
        value of the interaction on the central site

    Returns
    -------
    Vmat: numpy.ndarray (nsite,)
        Interaction matrix, here only a 1D array is needed.

    Raises
    ------
    ValueError
        raised when nsite is even
    """
    if nsite % 2 != 1:
        raise ValueError("ERROR: Number of sites has to be odd.")

    Vmat = np.zeros((nsite,))
    center = int((nsite - 1) / 2)

    Vmat[center] = U

    return Vmat


def get_1D_chain_nearest_neighbor_hopping_matrix(nsite: int, es: np.ndarray,
                                                 ts: np.ndarray,
                                                 boundary: bool = None
                                                 ) -> np.ndarray:
    """Return a hopping matrix of a tight binding chain with onsite energies
    'es', nearest neighbor hopping terms 'ts' and a boundary condition
    'boundary'.

    Parameters
    ----------
    nsite : int
        number of sites/orbitals in the system.

    es : numpy.ndarray (nsite,) or scalar
        onsite potentials

    ts : numpy.ndarray (nsite-1,) or scalar
        nearest neighbor hopping terms

    boundary : scalar, optional
         Coupling between first and last site in the chain, by default None

    Returns
    -------
    Tmat: numpy.ndarray (nsite,nsite)
        hopping matrix with tridiagonal structure
    """
    assert type(es) == type(ts)
    if type(ts) != np.ndarray:
        es = es * np.ones(nsite)
        ts = ts * np.ones(nsite - 1)

    Tmat = np.zeros((nsite, nsite))

    Tmat = np.diag(es) + np.diag(np.conj(ts), k=-1) + np.diag(ts, k=1)

    if boundary is not None:
        Tmat[0, - 1] = boundary
        Tmat[- 1, 0] = np.conj(boundary)

    return Tmat

###############################################################################


def hubbard_hamiltonian(T: np.ndarray, V: np.ndarray,
                        eop: Union[op.FermionicFockOperators, None] = None,
                        spinless: bool = False,
                        nelec: Union[int, None] = None
                        ) -> Union[Tuple[np.ndarray, np.ndarray],
                                   np.ndarray]:
    """Returns the Hamiltonian with on-site interaction and any geometry.

    Parameters
    ----------
    T : numpy.ndarray (dim,dim)
        Hopping matrix

    V : numpy.ndarray (dim,)
        On-site interaction strength. It can differ at each site

    eop : op.FermionicFockOperators, optional
        Fermionic Fock space, by default None. If not supplied a Fock space is
        generated within the function

    spinless : bool, optional
        Indicates if the fermionic system is spinless, by default False.
        Optional if a fermionic Fock space object is supplied or if 1/2 spin
        fermions.

    nelec : int, optional
        Particle number, by default None. If supplied the full Hamiltonian and
        the Hamiltonian in the 'nelec' particle subspace are returned

    Returns
    -------
    out: tuple or scipy.sparse.csc_matrix (dim,dim)
        If nelec is None, the full Fock space Hamiltonian returned.
        If nelec is an int, a tuple is returned, with the full Fock space
        Hamiltonian as first element and the Hamiltonian in the subspace with
        nelec electrons as second element.
    """
    if nelec == 0:
        return 0, 0  # or whatever value is correct for vacuum ...
    norbs = T.shape[0]
    if eop is not None:
        spinless = eop.spinless
    else:
        eop = op.FermionicFockOperators(norbs, spinless=spinless)
    assert eop.nsite == norbs

    T_mat = sparse.csc_matrix(
        (2**eop.spin_times_site, 2**eop.spin_times_site), dtype=np.complex64)
    V_mat = sparse.csc_matrix(
        (2**eop.spin_times_site, 2**eop.spin_times_site), dtype=np.complex64)

    for ii in range(norbs):
        if spinless:
            for jj in range(norbs):
                if abs(T[ii, jj]) > 1e-10:
                    T_mat += T[ii, jj] * (eop.cdag(ii).dot(eop.c(jj)))
        else:
            if abs(V[ii]) > 0:
                # Physicists' notation for V is assumed:
                # V[i,j,k,l]= cdag_i * cdag_j * c_l * c_k
                # only contributions with opposite spin
                V_mat += V[ii] * (eop.cdag(ii, "up").dot(
                    eop.cdag(ii, "do")).dot(eop.c(ii, "do").dot(eop.c(ii,
                                                                      "up"))))
                V_mat += V[ii] * (eop.cdag(ii, "do").dot(
                    eop.cdag(ii, "up")).dot(eop.c(ii, "up").dot(eop.c(ii,
                                                                      "do"))))

            for jj in range(norbs):
                if abs(T[ii, jj]) > 1e-10:
                    T_mat += T[ii, jj] * (eop.cdag(ii, "up").dot(eop.c(jj,
                                                                       "up")) +
                                          eop.cdag(ii, "do").dot(eop.c(jj,
                                                                       "do")))

    H_full = T_mat + 0.5 * V_mat

    if nelec is not None:
        H_part = H_full[eop.pascal_indices[nelec - 1]:eop.pascal_indices[
            nelec], eop.pascal_indices[nelec - 1]:eop.pascal_indices[nelec]]

        return H_full, H_part

    else:
        return H_full


def general_fermionic_hamiltonian(T: np.ndarray, V: np.ndarray,
                                  eop: Union[op.FermionicFockOperators,
                                             None] = None,
                                  spinless: bool = False,
                                  nelec: Union[int, None] = None
                                  ) -> Union[Tuple[np.ndarray, np.ndarray],
                                             np.ndarray]:
    """Returns the Hamiltonian with any kind of interaction and any geometry.

    Parameters
    ----------
    T : numpy.ndarray (dim,dim)
        Hopping matrix

    V : numpy.ndarray (dim,dim,dim,dim)
        Interaction strength matrix. It can differ at each site

    eop : op.FermionicFockOperators, optional
        Fermionic Fock space, by default None. If not supplied a Fock space is
        generated within the function

    spinless : bool, optional
        Indicates if the fermionic system is spinless, by default False.
        Optional if a fermionic Fock space object is supplied or if 1/2 spin
        fermions.

    nelec : int, optional
        Particle number, by default None. If supplied the full Hamiltonian and
        the Hamiltonian in the 'nelec' particle subspace are returned

    Returns
    -------
    out: tuple or scipy.sparse.csc_matrix (dim,dim)
        If nelec is None, the full Fock space Hamiltonian returned.
        If nelec is an int, a tuple is returned, with the full Fock space
        Hamiltonian as first element and the Hamiltonian in the subspace with
        nelec electrons as second element.
    """
    if nelec == 0:
        return 0, 0  # or whatever value is correct for vacuum ...

    norbs = T.shape[0]
    if eop is not None:
        spinless = eop.spinless
    else:
        eop = op.FermionicFockOperators(norbs, spinless=spinless)
    assert eop.nsite == norbs

    T_mat = sparse.csc_matrix(
        (2**eop.spin_times_site, 2**eop.spin_times_site), dtype=np.complex64)
    V_mat = sparse.csc_matrix(
        (2**eop.spin_times_site, 2**eop.spin_times_site), dtype=np.complex64)

    for ii in range(norbs):
        for jj in range(norbs):
            if abs(T[ii, jj]) > 1e-10:
                if spinless:
                    T_mat += T[ii, jj] * (eop.cdag(ii).dot(eop.c(jj)))
                else:
                    T_mat += T[ii, jj] * (eop.cdag(ii, "up").dot(eop.c(jj,
                                                                       "up")) +
                                          eop.cdag(ii, "do").dot(eop.c(jj,
                                                                       "do")))

            # Physicists' notation for V is assumed:
            # V[i,j,k,l]= cdag_i * cdag_j * c_l * c_k
            for kk in range(norbs):
                for ll in range(norbs):
                    if abs(V[ii, jj, kk, ll]) > 1e-10:
                        if ii == jj or kk == ll:
                            if not spinless:
                                # only contributions with opposite spin
                                V_mat += (V[ii, jj, kk, ll] *
                                          (eop.cdag(ii, "up").dot(
                                              eop.cdag(jj, "do")).dot(
                                              eop.c(ll, "do").dot(
                                                  eop.c(kk, "up")))))
                                V_mat += (V[ii, jj, kk, ll] *
                                          (eop.cdag(ii, "do").dot(
                                              eop.cdag(jj, "up")).dot(
                                              eop.c(ll, "up").dot(
                                                  eop.c(kk, "do")))))
                        # I definitely should look into symmetries of V-tensor
                        else:
                            if not spinless:
                                V_mat += (V[ii, jj, kk, ll] *
                                          (eop.cdag(ii, "up").dot(
                                              eop.cdag(jj, "up")).dot(
                                              eop.c(ll, "up").dot(
                                                  eop.c(kk, "up")))))
                                V_mat += (V[ii, jj, kk, ll] *
                                          (eop.cdag(ii, "do").dot(
                                              eop.cdag(jj, "do")).dot(
                                              eop.c(ll, "do").dot(
                                                  eop.c(kk, "do")))))
                                V_mat += (V[ii, jj, kk, ll] *
                                          (eop.cdag(ii, "up").dot(
                                              eop.cdag(jj, "do")).dot(
                                              eop.c(ll, "do").dot(
                                                  eop.c(kk, "up")))))
                                V_mat += (V[ii, jj, kk, ll] *
                                          (eop.cdag(ii, "do").dot(
                                              eop.cdag(jj, "up")).dot(
                                              eop.c(ll, "up").dot(
                                                  eop.c(kk, "do")))))
                            else:
                                V_mat += (V[ii, jj, kk, ll] *
                                          (eop.cdag(ii).dot(
                                              eop.cdag(jj)).dot(
                                              eop.c(ll).dot(
                                                  eop.c(kk)))))

    H_full = T_mat + 0.5 * V_mat
    if nelec is not None:
        H_part = H_full[eop.pascal_indices[nelec - 1]:eop.pascal_indices[
            nelec], eop.pascal_indices[nelec - 1]:eop.pascal_indices[nelec]]

        return H_full, H_part

    else:
        return H_full


if __name__ == "__main__":
    compare_hamiltonian = np.zeros(10)
    for i in range(10):
        e = np.random.rand()
        mu = np.random.rand()
        U = np.random.rand()
        t = np.array([[e - mu]])
        V = hubbard_U_center(1, U)
        H = hubbard_hamiltonian(t, V)

        compare_hamiltonian[i] = np.any(H == np.array([[0, 0, 0, 0],
                                                       [0, e - mu, 0, 0],
                                                       [0, 0, e - mu, 0],
                                                       [0, 0, 0,
                                                       U + 2. * (e - mu)]]))

    if not np.any(compare_hamiltonian):
        raise ValueError("ERROR: Atomic Hubbard model not correct.")
    print("Atomic Hubbard model correct")

    # Set up spinless Hamiltonian
    ts = np.array([1.0])
    Us = np.zeros(2)
    es = np.array([0, 0.5])
    nsite = 2
    Tmat = get_1D_chain_nearest_neighbor_hopping_matrix(nsite, es, ts)
    fermionic_op = op.FermionicFockOperators(nsite, spinless=True)

    H_spinless = hubbard_hamiltonian(
        Tmat, Us, eop=fermionic_op, spinless=True, nelec=1)
    print(H_spinless[0].toarray())
    print(H_spinless[1].toarray())
