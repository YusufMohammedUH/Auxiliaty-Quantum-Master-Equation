import pytest
import numpy as np
import src.auxiliary_mapping.auxiliary_system_parameter as auxp


def test_set_E_ph_symmetric_equal_value():
    ts = np.array([1, 2])
    es = np.array([1, 3])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    auxsys.set_E_ph_symmetric(es, ts)

    E = np.array([[1, 1, 0, 0, 0],
                  [1, 3, 2, 0, 0],
                  [0, 2, 0, 2, 0],
                  [0, 0, 2, -3, 1],
                  [0, 0, 0, 1, -1]])

    assert not (auxsys.E - E).any()


def test_set_E_ph_symmetric_rais_err_ts():
    ts = np.array([1, 2, 1])
    es = np.array([1, 3])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    with pytest.raises(AssertionError, match=r"ts doesn't have size Nb"):
        auxsys.set_E_ph_symmetric(es, ts)


def test_set_E_ph_symmetric_rais_err_es():
    ts = np.array([1, 2])
    es = np.array([1, 3, 2])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    with pytest.raises(AssertionError, match=r"es doesn't have size Nb"):
        auxsys.set_E_ph_symmetric(es, ts)


def test_set_E_general_equal_value():
    ts = np.array([1, 2, 2, 1])
    es = np.array([1, 3, 0, -3, -1])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    auxsys.set_E_general(es, ts)

    E = np.array([[1, 1, 0, 0, 0],
                  [1, 3, 2, 0, 0],
                  [0, 2, 0, 2, 0],
                  [0, 0, 2, -3, 1],
                  [0, 0, 0, 1, -1]])

    assert not (auxsys.E - E).any()


def test_set_E_general_rais_err_es():
    ts = np.array([1, 2, 2, 1])
    es = np.array([1, 3, 0, -3])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    with pytest.raises(AssertionError, match=r"es doesn't have size N"):
        auxsys.set_E_general(es, ts)


def test_set_E_general_rais_err_ts():
    ts = np.array([1, 2, 2, 1, 1])
    es = np.array([1, 3, 0, -3, -1])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    with pytest.raises(AssertionError, match=r"ts doesn't have size N-1"):
        auxsys.set_E_general(es, ts)


def test_get_Gamma_from_upper_tiagonal():
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)

    A = np.random.rand(3, 3).copy()
    A = np.triu(A, k=0).copy()
    B = auxsys.get_Gamma_from_upper_tiagonal(A)
    C = A.copy() + np.tril(A.conj().T, k=-1)
    assert not (B - C).any()


def test_get_Gamma_general_rais_err():
    gammas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    with pytest.raises(AssertionError):
        auxsys.get_Gamma_general(gammas)


def test_get_Gamma_general_value():
    gammas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
        1j * np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    Gamma_test = np.array([[1 + 1j, 2 + 2j, 0 + 0j, 3 + 3j, 4 + 4j],
                           [2 - 2j, 5 + 5j, 0 + 0j, 6 + 6j, 7 + 7j],
                           [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                           [3 - 3j, 6 - 6j, 0 + 0j, 8 + 8j, 9 + 9j],
                           [4 - 4j, 7 - 7j, 0 + 0j, 9 - 9j, 10 + 10j]])
    Gamma = auxsys.get_Gamma_general(gammas)

    assert not (Gamma - Gamma_test).any()


def test_get_Gamma2_ph_symmetric_value():
    ws = np.array([1])
    auxsys = auxp.AuxiliarySystem(2, ws)
    Gamma1 = np.array([[1 + 1j, 2 + 2j, 0 + 0j, 3 + 3j, 4 + 4j],
                      [2 - 2j, 5 + 5j, 0 + 0j, 6 + 6j, 7 + 7j],
                      [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                      [3 - 3j, 6 - 6j, 0 + 0j, 8 + 8j, 9 + 9j],
                      [4 - 4j, 7 - 7j, 0 + 0j, 9 - 9j, 10 + 10j]])

    Gamma2_test = np.array([[10. + 10.j, - 9. + 9.j, 0. + 0.j, - 7. + 7.j,
                             4. - 4.j],
                            [-9. - 9.j, 8. + 8.j, - 0. + 0.j, 6. - 6.j,
                             - 3. + 3.j],
                            [0. + 0.j, - 0. + 0.j, 0. + 0.j, - 0. + 0.j,
                             0. + 0.j],
                            [-7. - 7.j, 6. + 6.j, - 0. + 0.j, 5. + 5.j,
                             - 2. + 2.j], [4. + 4.j, - 3. - 3.j, 0. + 0.j,
                                           - 2. - 2.j, 1. + 1.j]])

    Gamma2 = auxsys.get_Gamma2_ph_symmetric(Gamma1)
    assert not (Gamma2 - Gamma2_test).any()


if __name__ == "__main__":
    pytest.main("-v test_auxiliary_hybridization_function.py")
