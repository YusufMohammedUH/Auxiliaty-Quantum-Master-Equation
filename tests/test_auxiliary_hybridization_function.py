import pytest
import sys
import numpy as np
sys.path.append('../src/')
import auxiliary_hybridization_function as auxhyb


def test_set_E_ph_symmetric():
    ts = np.array([1, 2])
    es = np.array([1, 3])

    auxsys = auxhyb.AuxiliarySystem(2)
    auxsys.set_E_ph_symmetric(es, ts)

    E = np.array([[1, 1, 0, 0, 0],
                  [1, 3, 2, 0, 0],
                  [0, 2, 0, 2, 0],
                  [0, 0, 2, -3, 1],
                  [0, 0, 0, 1, -1]])

    assert not (auxsys.E - E).any()


if __name__ == "__main__":
    pytest.main("-v test_auxiliary_hybridization_function.py")
