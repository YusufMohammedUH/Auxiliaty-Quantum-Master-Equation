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


class TestClassFermionicFockOperatorsSpinlessSortedParticleNumber:
    nsite = 2
    f_op = fock.FermionicFockOperators(nsite, spinless=True)
    identity = sparse.eye(2**f_op.spin_times_site, dtype=complex)

    def test_FermionicFockOperators_anticommutation(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # purely fock creation and annihilation operators
                anti_commutation_c_cdag = (self.f_op.cdag(i)
                                           * self.f_op.c(j)
                                           + self.f_op.c(j)
                                           * self.f_op.cdag(i))
                anti_commutation_c_c = (self.f_op.c(i)
                                        * self.f_op.c(j)
                                        + self.f_op.c(j)
                                        * self.f_op.c(i))
                anti_commutation_cdag_cdag = (self.f_op.cdag(i)
                                              * self.f_op.cdag(j)
                                              + self.f_op.cdag(j)
                                              * self.f_op.cdag(i))
                if i == j:
                    assert (anti_commutation_c_cdag -
                            self.identity).count_nonzero() == 0

                else:
                    assert (
                        anti_commutation_c_cdag).count_nonzero() == 0
                    assert (anti_commutation_c_c).count_nonzero() == 0
                    assert (
                        anti_commutation_cdag_cdag).count_nonzero() == 0

    def test_FermionicFockOperators_particle_number_sorted(self):
        assert np.all(self.f_op.N.diagonal() == np.sort(
            self.f_op.N.diagonal()))

    def test_FermionicFockOperators_pascal_indices_spinless(self):
        pascal_indices_accumulated = 0
        for i in range(self.f_op.spin_times_site + 1):
            pascal_indices_accumulated += (self.f_op.N.diagonal() == i).sum()
            print(self.f_op.pascal_indices[i],
                  "  ", pascal_indices_accumulated)
            assert self.f_op.pascal_indices[i] == pascal_indices_accumulated


class TestClassFermionicFockOperatorsSpinlessUnsortedParticleNumber:
    nsite = 2
    f_op = fock.FermionicFockOperators(nsite, spinless=True,
                                       sorted_particle_number=False)
    identity = sparse.eye(2**f_op.spin_times_site, dtype=complex)

    def test_FermionicFockOperators_anticommutation(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # purely fock creation and annihilation operators
                anti_commutation_c_cdag = (self.f_op.cdag(i)
                                           * self.f_op.c(j)
                                           + self.f_op.c(j)
                                           * self.f_op.cdag(i))
                anti_commutation_c_c = (self.f_op.c(i)
                                        * self.f_op.c(j)
                                        + self.f_op.c(j)
                                        * self.f_op.c(i))
                anti_commutation_cdag_cdag = (self.f_op.cdag(i)
                                              * self.f_op.cdag(j)
                                              + self.f_op.cdag(j)
                                              * self.f_op.cdag(i))
                if i == j:
                    assert (anti_commutation_c_cdag -
                            self.identity).count_nonzero() == 0

                else:
                    assert (
                        anti_commutation_c_cdag).count_nonzero() == 0
                    assert (anti_commutation_c_c).count_nonzero() == 0
                    assert (
                        anti_commutation_cdag_cdag).count_nonzero() == 0

    def test_FermionicFockOperators_particle_number_unsorted(self):
        if (self.f_op.P - self.identity).nnz == 0:
            assert np.all(self.f_op.N.diagonal() == np.sort(
                self.f_op.N.diagonal()))
        else:
            assert not np.all(self.f_op.N.diagonal() == np.sort(
                self.f_op.N.diagonal()))

    def test_FermionicFockOperators_pascal_indices_spinless(self):
        pascal_indices_accumulated = 0
        for i in range(self.f_op.spin_times_site + 1):
            pascal_indices_accumulated += (self.f_op.N.diagonal() == i).sum()
            print(self.f_op.pascal_indices[i],
                  "  ", pascal_indices_accumulated)
            assert self.f_op.pascal_indices[i] == pascal_indices_accumulated


class TestClassFermionicFockOperatorsSpinfulSortedParticleNumber:
    nsite = 2
    f_op = fock.FermionicFockOperators(nsite)
    identity = sparse.eye(2**f_op.spin_times_site, dtype=complex)

    def test_FermionicFockOperators_anticommutation(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # purely fock creation and annihilation operators
                        anti_commutation_c_cdag = (self.f_op.cdag(i, s1)
                                                   * self.f_op.c(j, s2)
                                                   + self.f_op.c(j, s2)
                                                   * self.f_op.cdag(i, s1))
                        anti_commutation_c_c = (self.f_op.c(i, s1)
                                                * self.f_op.c(j, s2)
                                                + self.f_op.c(j, s2)
                                                * self.f_op.c(i, s1))
                        anti_commutation_cdag_cdag = (self.f_op.cdag(i, s1)
                                                      * self.f_op.cdag(j, s2)
                                                      + self.f_op.cdag(j, s2)
                                                      * self.f_op.cdag(i, s1))
                        if i == j and s1 == s2:
                            assert (anti_commutation_c_cdag -
                                    self.identity).count_nonzero() == 0

                        else:
                            assert (
                                anti_commutation_c_cdag).count_nonzero() == 0
                            assert (anti_commutation_c_c).count_nonzero() == 0
                            assert (
                                anti_commutation_cdag_cdag).count_nonzero() == 0

    def test_FermionicFockOperators_particle_number_sorted(self):
        assert np.all(self.f_op.N.diagonal() == np.sort(
            self.f_op.N.diagonal()))

    def test_FermionicFockOperators_pascal_indices(self):
        pascal_indices_accumulated = 0
        for i in range(self.f_op.spin_times_site + 1):
            pascal_indices_accumulated += (self.f_op.N.diagonal() == i).sum()
            print(self.f_op.pascal_indices[i],
                  "  ", pascal_indices_accumulated)
            assert self.f_op.pascal_indices[i] == pascal_indices_accumulated


class TestClassFermionicFockOperatorsSpinfulUnortedParticleNumber:
    nsite = 2
    f_op = fock.FermionicFockOperators(nsite, sorted_particle_number=False)
    identity = sparse.eye(2**f_op.spin_times_site, dtype=complex)

    def test_FermionicFockOperators_anticommutation(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # purely fock creation and annihilation operators
                        anti_commutation_c_cdag = (self.f_op.cdag(i, s1)
                                                   * self.f_op.c(j, s2)
                                                   + self.f_op.c(j, s2)
                                                   * self.f_op.cdag(i, s1))
                        anti_commutation_c_c = (self.f_op.c(i, s1)
                                                * self.f_op.c(j, s2)
                                                + self.f_op.c(j, s2)
                                                * self.f_op.c(i, s1))
                        anti_commutation_cdag_cdag = (self.f_op.cdag(i, s1)
                                                      * self.f_op.cdag(j, s2)
                                                      + self.f_op.cdag(j, s2)
                                                      * self.f_op.cdag(i, s1))
                        if i == j and s1 == s2:
                            assert (anti_commutation_c_cdag -
                                    self.identity).count_nonzero() == 0

                        else:
                            assert (
                                anti_commutation_c_cdag).count_nonzero() == 0
                            assert (anti_commutation_c_c).count_nonzero() == 0
                            assert (
                                anti_commutation_cdag_cdag).count_nonzero() == 0

    def test_FermionicFockOperators_particle_number_unsorted(self):
        if (self.f_op.P - self.identity).nnz == 0:
            assert np.all(self.f_op.N.diagonal() == np.sort(
                self.f_op.N.diagonal()))
        else:
            assert not np.all(self.f_op.N.diagonal() == np.sort(
                self.f_op.N.diagonal()))

    def test_FermionicFockOperators_pascal_indices(self):
        pascal_indices_accumulated = 0
        for i in range(self.f_op.spin_times_site + 1):
            pascal_indices_accumulated += (self.f_op.N.diagonal() == i).sum()
            print(self.f_op.pascal_indices[i],
                  "  ", pascal_indices_accumulated)
            assert self.f_op.pascal_indices[i] == pascal_indices_accumulated


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
