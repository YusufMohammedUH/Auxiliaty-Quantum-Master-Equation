import pytest
import numpy as np
import src.define_fock_space_operators as fock
from scipy import sparse


def test_commutator():
    n = 100
    A = sparse.random(n, n, density=0.03, format='csc')
    B = sparse.random(n, n, density=0.03, format='csc')
    assert np.all((fock.commutator(A, B).todense()
                  == (A * B - B * A.todense())))


def test_anti_commutator():
    n = 100
    A = sparse.random(n, n, density=0.03, format='csc')
    B = sparse.random(n, n, density=0.03, format='csc')
    assert np.all((fock.anti_commutator(A, B).todense()
                  == (A * B + B * A.todense())))


def test_FermionicFockOperators_spinfull_anticommutation():
    nsite = 2
    f_op = fock.FermionicFockOperators(nsite)
    identity = sparse.eye(2**f_op.spin_times_site, dtype=complex)
    for i in range(nsite):
        for j in range(nsite):
            for s1 in ["up", "do"]:
                for s2 in ["up", "do"]:
                    # purely fock creation and annihilation operators
                    anti_commutation_c_cdag = (f_op.cdag(i, s1)
                                               * f_op.c(j, s2)
                                               + f_op.c(j, s2)
                                               * f_op.cdag(i, s1))
                    anti_commutation_c_c = (f_op.c(i, s1)
                                            * f_op.c(j, s2)
                                            + f_op.c(j, s2)
                                            * f_op.c(i, s1))
                    anti_commutation_cdag_cdag = (f_op.cdag(i, s1)
                                                  * f_op.cdag(j, s2)
                                                  + f_op.cdag(j, s2)
                                                  * f_op.cdag(i, s1))
                    if i == j and s1 == s2:
                        assert (anti_commutation_c_cdag -
                                identity).count_nonzero() == 0

                    else:
                        assert (
                            anti_commutation_c_cdag).count_nonzero() == 0
                        assert (anti_commutation_c_c).count_nonzero() == 0
                        assert (
                            anti_commutation_cdag_cdag).count_nonzero() == 0


def test_FermionicFockOperators_spinless_anticommutation():
    nsite = 4
    f_op = fock.FermionicFockOperators(nsite, spinless=True)
    identity = sparse.eye(2**f_op.spin_times_site, dtype=complex)
    for i in range(nsite):
        for j in range(nsite):
            # purely fock creation and annihilation operators
            anti_commutation_c_cdag = (f_op.cdag(i)
                                       * f_op.c(j)
                                       + f_op.c(j)
                                       * f_op.cdag(i))
            anti_commutation_c_c = (f_op.c(i)
                                    * f_op.c(j)
                                    + f_op.c(j)
                                    * f_op.c(i))
            anti_commutation_cdag_cdag = (f_op.cdag(i)
                                          * f_op.cdag(j)
                                          + f_op.cdag(j)
                                          * f_op.cdag(i))
            if i == j:
                assert (anti_commutation_c_cdag -
                        identity).count_nonzero() == 0

            else:
                assert (
                    anti_commutation_c_cdag).count_nonzero() == 0
                assert (anti_commutation_c_c).count_nonzero() == 0
                assert (
                    anti_commutation_cdag_cdag).count_nonzero() == 0


def test_FermionicFockOperators_particle_number_sorted():
    nsite = 3
    f_op = fock.FermionicFockOperators(nsite, spinless=True)
    assert np.all(f_op.N.diagonal() == np.sort(f_op.N.diagonal()))


def test_FermionicFockOperators_pascal_indices_spinfull():
    nsite = 3
    f_op = fock.FermionicFockOperators(nsite)
    pascal_indices_accumulated = 0
    for i in range(f_op.spin_times_site + 1):
        pascal_indices_accumulated += (f_op.N.diagonal() == i).sum()
        print(f_op.pascal_indices[i], "  ", pascal_indices_accumulated)
        assert f_op.pascal_indices[i] == pascal_indices_accumulated


def test_FermionicFockOperators_pascal_indices_spinless():
    nsite = 3
    f_op = fock.FermionicFockOperators(nsite, spinless=True)
    pascal_indices_accumulated = 0
    for i in range(f_op.spin_times_site + 1):
        pascal_indices_accumulated += (f_op.N.diagonal() == i).sum()
        print(f_op.pascal_indices[i], "  ", pascal_indices_accumulated)
        assert f_op.pascal_indices[i] == pascal_indices_accumulated


def test_BosonicFockOperators_commutation():
    nmodes = 3
    nb_max = 5
    b_op = fock.BosonicFockOperators(nmodes, nb_max)

    for i in range(nmodes):
        for j in range(nmodes):
            # purely fock creation and annihilation operators
            commutation_b_bdag = (b_op.b(j)
                                  * b_op.bdag(i)
                                  - b_op.bdag(i)
                                  * b_op.b(j))
            commutation_b_b = (b_op.b(i)
                                    * b_op.b(j)
                                    - b_op.b(j)
                                    * b_op.b(i))
            commutation_bdag_bdag = (b_op.bdag(i)
                                     * b_op.bdag(j)
                                     - b_op.bdag(j)
                                     * b_op.bdag(i))
            if i == j:
                assert (np.isclose(commutation_b_bdag.diagonal(), -
                                   nb_max * np.ones((b_op.nb_max**b_op.nmodes,)
                                                    ))).sum() == \
                    b_op.nb_max**(b_op.nmodes - 1)
                assert (np.isclose(commutation_b_bdag.diagonal(),
                                   np.ones((b_op.nb_max**b_op.nmodes,)
                                           ))).sum() == \
                    b_op.nb_max**(b_op.nmodes) - b_op.nb_max**(b_op.nmodes - 1)

            else:
                assert (
                    commutation_b_bdag).count_nonzero() == 0
                assert (commutation_b_b).count_nonzero() == 0
                assert (
                    commutation_bdag_bdag).count_nonzero() == 0


if __name__ == "__main__":

    pytest.main("-v test_frequency_greens_function.py")
