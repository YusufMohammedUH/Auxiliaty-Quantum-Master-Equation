import pytest
import numpy as np
import src.hilber_space.model_hamiltonian as ham
import src.hilber_space.define_fock_space_operators as op
import random


def test_hubbard_U_center():
    n = 2 * random.randint(1, 5) + 1
    U = random.random()
    Us = np.zeros(n)
    Us[int((n - 1) / 2)] = U
    assert np.all(ham.hubbard_U_center(n, U) == Us)


def test_get_1D_chain_nearest_neighbor_hopping_matrix_build_from_vector():
    n = random.randint(1, 10)
    es = np.random.rand(n)
    ts = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)
    boundary = random.random()
    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        n, es, ts, boundary)

    Tmat = np.zeros((n, n))

    Tmat = np.diag(es) + np.diag(np.conj(ts), k=-1) + np.diag(ts, k=1)

    Tmat[0, - 1] = boundary
    Tmat[- 1, 0] = np.conj(boundary)

    assert np.all(T_mat == Tmat)


def test_get_1D_chain_nearest_neighbor_hopping_matrix_build_from_constants():
    n = random.randint(1, 10)

    es = (random.random() + 1j * random.random())
    ts = (random.random() + 1j * random.random())
    boundary = random.random()
    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        n, es, ts, boundary)

    Tmat = np.zeros((n, n))

    es = (es) * np.ones(n)
    ts = (ts) * np.ones(n - 1)
    Tmat = np.diag(es) + np.diag(np.conj(ts), k=-1) + np.diag(ts, k=1)

    Tmat[0, - 1] = boundary
    Tmat[- 1, 0] = np.conj(boundary)

    assert np.all(T_mat == Tmat)


def test_get_1D_chain_nearest_neighbor_hopping_matrix_self_adjoint():
    for n in range(1, 10):
        es = np.random.rand(n)
        ts = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)
        boundary = random.random()
        T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
            n, es, ts, boundary)
        assert np.all(T_mat == T_mat.T.conj())


def test_hubbard_hamiltonian_single_site():
    for i in range(10):
        e = np.random.rand()
        mu = np.random.rand()
        U = np.random.rand()
        t = np.array([[e - mu]])
        V = ham.hubbard_U_center(1, U)
        H = ham.hubbard_hamiltonian(t, V)

        assert np.all(H == np.array([[0, 0, 0, 0],
                                     [0, e - mu, 0, 0],
                                     [0, 0, e - mu, 0],
                                     [0, 0, 0,
                                      U + 2. * (e - mu)]]))


def test_hubbard_hamiltonian_self_adjoint():
    for n in range(1, 4):
        es = 1. + 0.j
        ts = random.random() + 1j * random.random()
        V = np.random.rand(n)
        T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
            n, es, ts)
        fermionic_op = op.FermionicFockOperators(n)

        H = ham.hubbard_hamiltonian(T_mat, V, eop=fermionic_op)

        assert np.all(H.todense() == H.todense().T.conj())


def test_hubbard_hamiltonian_two_site_no_interaction_explicit():
    n = 2
    es = 1. + 0.j
    ts = random.random() + 1j * random.random()
    V = np.zeros(n)
    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        n, es, ts)
    fermionic_op = op.FermionicFockOperators(n)

    H = ham.hubbard_hamiltonian(T_mat, V, eop=fermionic_op)

    H_test = np.zeros(H.shape, dtype=np.complex64)
    H_test[1, 3] = H_test[2, 4] = H_test[5, 8] = H_test[8, 10] = ts
    H_test[7, 10] = H_test[5, 7] = H_test[11, 13] = H_test[12, 14] = -ts
    H_test = H_test + H_test.conj().T + fermionic_op.N

    assert np.all(H.todense() == H_test)


def test_hubbard_hamiltonian_two_site_interaction():
    n = 2
    es = 0. + 0.j
    ts = 0. + 0.j
    V = random.random() * np.ones(n)
    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        n, es, ts)
    fermionic_op = op.FermionicFockOperators(n)

    H = ham.hubbard_hamiltonian(T_mat, V, eop=fermionic_op)
    H_diag = np.zeros((2 ** fermionic_op.spin_times_site,))
    H_diag[5] = H_diag[10] = H_diag[11] = H_diag[12] = H_diag[13] = \
        H_diag[14] = V[0]
    H_diag[-1] = 2 * V[0]

    assert np.all(H == np.diag(H_diag))


def test_general_fermionic_hamiltonian_single_site():
    for i in range(10):
        e = np.random.rand()
        mu = np.random.rand()
        U = np.random.rand()
        t = np.array([[e - mu]])
        V = U * np.ones((1, 1, 1, 1))
        H = ham.general_fermionic_hamiltonian(t, V)

        assert np.all(H == np.array([[0, 0, 0, 0],
                                     [0, e - mu, 0, 0],
                                     [0, 0, e - mu, 0],
                                     [0, 0, 0,
                                      U + 2. * (e - mu)]]))


def test_general_fermionic_hamiltonian_self_adjoint():
    for n in range(1, 4):
        es = 1. + 0.j
        ts = random.random() + 1j * random.random()
        V = np.zeros((n, n, n, n))
        d = np.einsum('ijji->ij', V)
        A = np.random.rand(n, n)
        d[:] = A + A.T.conj()
        T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
            n, es, ts)
        fermionic_op = op.FermionicFockOperators(n)

        H = ham.general_fermionic_hamiltonian(T_mat, V, eop=fermionic_op)

        assert np.all(H.todense() == H.todense().T.conj())


def test_general_fermionic_hamiltonian_two_site_no_interaction_explicit():
    n = 2
    es = 1. + 0.j
    ts = random.random() + 1j * random.random()
    V = np.zeros((n, n, n, n))
    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        n, es, ts)
    fermionic_op = op.FermionicFockOperators(n)

    H = ham.general_fermionic_hamiltonian(T_mat, V, eop=fermionic_op)

    H_test = np.zeros(H.shape, dtype=np.complex64)
    H_test[1, 3] = H_test[2, 4] = H_test[5, 8] = H_test[8, 10] = ts
    H_test[7, 10] = H_test[5, 7] = H_test[11, 13] = H_test[12, 14] = -ts
    H_test = H_test + H_test.conj().T + fermionic_op.N

    assert np.all(H.todense() == H_test)


def test_general_fermionic_hamiltonian_two_site_interaction():
    n = 2
    es = 0 + 0j
    ts = 0 + 0j
    V = np.zeros((n, n, n, n))
    d = np.einsum('ijji->ij', V)
    A = np.random.rand(n, n)
    A = A + A.T.conj()
    d[:] = A

    T_mat = ham.get_1D_chain_nearest_neighbor_hopping_matrix(
        n, es, ts)
    fermionic_op = op.FermionicFockOperators(n)

    H = ham.general_fermionic_hamiltonian(T_mat, V, eop=fermionic_op)
    H_explicit = np.zeros(
        (2 ** fermionic_op.spin_times_site, 2 ** fermionic_op.spin_times_site))
    H_explicit[5, 5] = A[0, 0]
    H_explicit[10, 10] = A[1, 1]
    H_explicit[11, 11] = H_explicit[12, 12] = A[0, 0] - A[0, 1]
    H_explicit[13, 13] = H_explicit[14, 14] = A[1, 1] - A[0, 1]
    H_explicit[15, 15] = (A[0, 0] + A[1, 1] - 2 * A[0, 1])
    H_explicit[6, 6] = H_explicit[9, 9] = -A[0, 1]
    H_explicit[7, 8] = H_explicit[8, 7] = -A[0, 1]

    assert np.all(np.isclose((H.todense() - H_explicit), np.zeros(
        (2 ** fermionic_op.spin_times_site, 2 ** fermionic_op.spin_times_site))
    ))


if __name__ == "__main__":
    pytest.main("-v test_frequency_greens_function.py")
