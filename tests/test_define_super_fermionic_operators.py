import pytest
from scipy import sparse
import src.super_fermionic_space.define_super_fermionic_operators as sf_op
import src.hilber_space.define_fock_space_operators as fop


class TestClassSuperFermionicOperatorsNoComplexPhaseSpinless:
    nsite = 2
    spinless = True
    fl_op = sf_op.SuperFermionicOperators(nsite, spinless=spinless,
                                          tilde_conjugationrule_phase=False)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=complex, format="csc")

    def test_commutation_relation_fock_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # purely fock creation and annihilation operators
                anti_commutation_fock_c_cdag = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.cdag(j))
                anti_commutation_fock_c_c = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.c(j))
                anti_commutation_fock_cdag_cdag = \
                    fop.anti_commutator(
                        self.fl_op.cdag(i),
                        self.fl_op.cdag(j))

                if i == j:
                    assert (anti_commutation_fock_c_cdag -
                            self.identity_super_fermionic).nnz == 0
                    assert anti_commutation_fock_c_c.nnz == 0
                    assert anti_commutation_fock_cdag_cdag.nnz == 0

                else:
                    assert anti_commutation_fock_c_cdag.nnz == 0
                    assert anti_commutation_fock_c_c.nnz == 0
                    assert anti_commutation_fock_cdag_cdag.nnz == 0

    def test_commutation_relation_tilde_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # purely tilde creation and annihilation operators
                anti_commutation_tilde_c_cdag = \
                    fop.anti_commutator(
                        self.fl_op.c_tilde(i),
                        self.fl_op.cdag_tilde(j))
                anti_commutation_tilde_c_c =  \
                    fop.anti_commutator(
                        self.fl_op.c_tilde(i),
                        self.fl_op.c_tilde(j))
                anti_commutation_tilde_cdag_cdag = \
                    fop.anti_commutator(
                        self.fl_op.cdag_tilde(i),
                        self.fl_op.cdag_tilde(j))

                if i == j:
                    assert (anti_commutation_tilde_c_cdag -
                            self.identity_super_fermionic).nnz == 0
                    assert anti_commutation_tilde_c_c.nnz == 0
                    assert anti_commutation_tilde_cdag_cdag.nnz == 0

                else:
                    assert anti_commutation_tilde_c_cdag.nnz == 0
                    assert anti_commutation_tilde_c_c.nnz == 0
                    assert anti_commutation_tilde_cdag_cdag.nnz == 0

    def test_commutation_relation_mixed_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # mixed creation and annihilation operators
                anti_commutation_mixed_c_cdag_tilde = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.cdag_tilde(j))
                anti_commutation_mixed_c_c_tilde = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.c_tilde(j))
                anti_commutation_mixed_cdag_cdag_tilde = \
                    fop.anti_commutator(
                        self.fl_op.cdag(i), self.fl_op.cdag_tilde(j))

                if (not i == j):
                    assert anti_commutation_mixed_c_cdag_tilde.nnz == 0
                    assert anti_commutation_mixed_c_c_tilde.nnz == 0
                    assert anti_commutation_mixed_cdag_cdag_tilde.nnz == 0

    def test_fock_tilde_operator_correspondence(self):
        n_nonzeros = 0
        for i in range(self.nsite):
            n_nonzeros += (self.fl_op.c(i)
                           - self.fl_op.cdag_tilde(i)
                           ).dot(self.fl_op.left_vacuum).nnz

            n_nonzeros += (self.fl_op.cdag(i)
                           + self.fl_op.c_tilde(i)
                           ).dot(self.fl_op.left_vacuum).nnz
        assert n_nonzeros == 0

    def test_fock_tilde_operator_correspondence_pair(self):
        for i in range(self.nsite):
            for j in range(self.nsite):

                cdag_c = (self.fl_op.cdag(i)
                          * self.fl_op.c(j))

                c_c = (self.fl_op.c(i)
                       * self.fl_op.c(j))

                cdag_c_tilde = (
                    self.fl_op.cdag_tilde(j)
                    * self.fl_op.c_tilde(i))

                c_c_tilde = (
                    self.fl_op.c_tilde(j)
                    * self.fl_op.c_tilde(i))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_pair_get_super_fermi_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):

                cdag_c = self.fl_op.get_super_fermionic_operator(
                    self.fl_op.fock_ops.cdag(i)
                    * self.fl_op.fock_ops.c(j))

                c_c = self.fl_op.get_super_fermionic_operator(
                    self.fl_op.fock_ops.c(i)
                    * self.fl_op.fock_ops.c(j))

                cdag_c_tilde = \
                    self.fl_op.get_super_fermionic_tilde_operator(
                        self.fl_op.fock_ops.cdag(j)
                        * self.fl_op.fock_ops.c(i))

                c_c_tilde = self.fl_op.get_super_fermionic_tilde_operator(
                    self.fl_op.fock_ops.c(j)
                    * self.fl_op.fock_ops.c(i))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        cdag_cdag_c_c = (
                            self.fl_op.cdag(i)
                            * self.fl_op.cdag(j)
                            * self.fl_op.c(k)
                            * self.fl_op.c(l))

                        cdag_c_cdag_c = (
                            self.fl_op.cdag(i)
                            * self.fl_op.c(j)
                            * self.fl_op.cdag(k)
                            * self.fl_op.c(l))

                        cdag_c_c_cdag = (
                            self.fl_op.cdag(i)
                            * self.fl_op.c(j)
                            * self.fl_op.c(k)
                            * self.fl_op.cdag(l))

                        cdag_cdag_c_c_tilde = (
                            self.fl_op.cdag_tilde(l)
                            * self.fl_op.cdag_tilde(k)
                            * self.fl_op.c_tilde(j)
                            * self.fl_op.c_tilde(i))

                        cdag_c_cdag_c_tilde = (
                            self.fl_op.cdag_tilde(l)
                            * self.fl_op.c_tilde(k)
                            * self.fl_op.cdag_tilde(j)
                            * self.fl_op.c_tilde(i))

                        cdag_c_c_cdag_tilde = (
                            self.fl_op.c_tilde(l)
                            * self.fl_op.cdag_tilde(k)
                            * self.fl_op.cdag_tilde(j)
                            * self.fl_op.c_tilde(i))

        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_cdag_c_c.count_nonzero() == 0

        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_cdag_c.count_nonzero() == 0

        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_c_cdag.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four_get_super_fermi_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        cdag_cdag_c_c = \
                            self.fl_op.get_super_fermionic_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.cdag(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_cdag_c = \
                            self.fl_op.get_super_fermionic_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.cdag(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_c_cdag = \
                            self.fl_op.get_super_fermionic_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.cdag(l))

                        cdag_cdag_c_c_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.cdag(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_cdag_c_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.cdag(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_c_cdag_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.cdag(l))

                        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
                            self.fl_op.left_vacuum) - \
                            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
                        assert correspondence_cdag_cdag_c_c.count_nonzero(
                        ) == 0

                        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
                            self.fl_op.left_vacuum) - \
                            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
                        assert correspondence_cdag_c_cdag_c.count_nonzero(
                        ) == 0

                        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
                            self.fl_op.left_vacuum) - \
                            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
                        assert correspondence_cdag_c_c_cdag.count_nonzero(
                        ) == 0


class TestClassSuperFermionicOperatorsComplexPhaseSpinless:
    nsite = 2
    spinless = True
    fl_op = sf_op.SuperFermionicOperators(nsite, spinless=spinless,
                                          tilde_conjugationrule_phase=True)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=complex, format="csc")

    def test_commutation_relation_fock_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # purely fock creation and annihilation operators
                anti_commutation_fock_c_cdag = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.cdag(j))
                anti_commutation_fock_c_c = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.c(j))
                anti_commutation_fock_cdag_cdag = \
                    fop.anti_commutator(
                        self.fl_op.cdag(i),
                        self.fl_op.cdag(j))

                if i == j:
                    assert (anti_commutation_fock_c_cdag -
                            self.identity_super_fermionic).nnz == 0
                    assert anti_commutation_fock_c_c.nnz == 0
                    assert anti_commutation_fock_cdag_cdag.nnz == 0

                else:
                    assert anti_commutation_fock_c_cdag.nnz == 0
                    assert anti_commutation_fock_c_c.nnz == 0
                    assert anti_commutation_fock_cdag_cdag.nnz == 0

    def test_commutation_relation_tilde_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # purely tilde creation and annihilation operators
                anti_commutation_tilde_c_cdag = \
                    fop.anti_commutator(
                        self.fl_op.c_tilde(i),
                        self.fl_op.cdag_tilde(j))
                anti_commutation_tilde_c_c =  \
                    fop.anti_commutator(
                        self.fl_op.c_tilde(i),
                        self.fl_op.c_tilde(j))
                anti_commutation_tilde_cdag_cdag = \
                    fop.anti_commutator(
                        self.fl_op.cdag_tilde(i),
                        self.fl_op.cdag_tilde(j))

                if i == j:
                    assert (anti_commutation_tilde_c_cdag -
                            self.identity_super_fermionic).nnz == 0
                    assert anti_commutation_tilde_c_c.nnz == 0
                    assert anti_commutation_tilde_cdag_cdag.nnz == 0

                else:
                    assert anti_commutation_tilde_c_cdag.nnz == 0
                    assert anti_commutation_tilde_c_c.nnz == 0
                    assert anti_commutation_tilde_cdag_cdag.nnz == 0

    def test_commutation_relation_mixed_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                # mixed creation and annihilation operators
                anti_commutation_mixed_c_cdag_tilde = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.cdag_tilde(j))
                anti_commutation_mixed_c_c_tilde = \
                    fop.anti_commutator(self.fl_op.c(i),
                                        self.fl_op.c_tilde(j))
                anti_commutation_mixed_cdag_cdag_tilde = \
                    fop.anti_commutator(
                        self.fl_op.cdag(i), self.fl_op.cdag_tilde(j))

                if (not i == j):
                    assert anti_commutation_mixed_c_cdag_tilde.nnz == 0
                    assert anti_commutation_mixed_c_c_tilde.nnz == 0
                    assert anti_commutation_mixed_cdag_cdag_tilde.nnz == 0

    def test_fock_tilde_operator_correspondence(self):
        n_nonzeros = 0
        for i in range(self.nsite):
            n_nonzeros += (self.fl_op.c(i)
                           + 1j * self.fl_op.cdag_tilde(i)
                           ).dot(self.fl_op.left_vacuum).nnz

            n_nonzeros += (self.fl_op.cdag(i)
                           + 1j * self.fl_op.c_tilde(i)
                           ).dot(self.fl_op.left_vacuum).nnz
        assert n_nonzeros == 0

    def test_fock_tilde_operator_correspondence_pair(self):
        for i in range(self.nsite):
            for j in range(self.nsite):

                cdag_c = (self.fl_op.cdag(i)
                          * self.fl_op.c(j))

                c_c = (self.fl_op.c(i)
                       * self.fl_op.c(j))

                cdag_c_tilde = (
                    self.fl_op.cdag_tilde(j)
                    * self.fl_op.c_tilde(i))

                c_c_tilde = (
                    self.fl_op.c_tilde(j)
                    * self.fl_op.c_tilde(i))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_pair_get_super_fermi_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):

                cdag_c = self.fl_op.get_super_fermionic_operator(
                    self.fl_op.fock_ops.cdag(i)
                    * self.fl_op.fock_ops.c(j))

                c_c = self.fl_op.get_super_fermionic_operator(
                    self.fl_op.fock_ops.c(i)
                    * self.fl_op.fock_ops.c(j))

                cdag_c_tilde = \
                    self.fl_op.get_super_fermionic_tilde_operator(
                        self.fl_op.fock_ops.cdag(j)
                        * self.fl_op.fock_ops.c(i))

                c_c_tilde = self.fl_op.get_super_fermionic_tilde_operator(
                    self.fl_op.fock_ops.c(j)
                    * self.fl_op.fock_ops.c(i))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        cdag_cdag_c_c = (
                            self.fl_op.cdag(i)
                            * self.fl_op.cdag(j)
                            * self.fl_op.c(k)
                            * self.fl_op.c(l))

                        cdag_c_cdag_c = (
                            self.fl_op.cdag(i)
                            * self.fl_op.c(j)
                            * self.fl_op.cdag(k)
                            * self.fl_op.c(l))

                        cdag_c_c_cdag = (
                            self.fl_op.cdag(i)
                            * self.fl_op.c(j)
                            * self.fl_op.c(k)
                            * self.fl_op.cdag(l))

                        cdag_cdag_c_c_tilde = (
                            self.fl_op.cdag_tilde(l)
                            * self.fl_op.cdag_tilde(k)
                            * self.fl_op.c_tilde(j)
                            * self.fl_op.c_tilde(i))

                        cdag_c_cdag_c_tilde = (
                            self.fl_op.cdag_tilde(l)
                            * self.fl_op.c_tilde(k)
                            * self.fl_op.cdag_tilde(j)
                            * self.fl_op.c_tilde(i))

                        cdag_c_c_cdag_tilde = (
                            self.fl_op.c_tilde(l)
                            * self.fl_op.cdag_tilde(k)
                            * self.fl_op.cdag_tilde(j)
                            * self.fl_op.c_tilde(i))

        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_cdag_c_c.count_nonzero() == 0

        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_cdag_c.count_nonzero() == 0

        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_c_cdag.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four_get_super_fermi_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        cdag_cdag_c_c = \
                            self.fl_op.get_super_fermionic_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.cdag(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_cdag_c = \
                            self.fl_op.get_super_fermionic_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.cdag(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_c_cdag = \
                            self.fl_op.get_super_fermionic_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.cdag(l))

                        cdag_cdag_c_c_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.cdag(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_cdag_c_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.cdag(k)
                                * self.fl_op.fock_ops.c(l))

                        cdag_c_c_cdag_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(i)
                                * self.fl_op.fock_ops.c(j)
                                * self.fl_op.fock_ops.c(k)
                                * self.fl_op.fock_ops.cdag(l))

                        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
                            self.fl_op.left_vacuum) - \
                            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
                        assert correspondence_cdag_cdag_c_c.count_nonzero(
                        ) == 0

                        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
                            self.fl_op.left_vacuum) - \
                            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
                        assert correspondence_cdag_c_cdag_c.count_nonzero(
                        ) == 0

                        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
                            self.fl_op.left_vacuum) - \
                            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
                        assert correspondence_cdag_c_c_cdag.count_nonzero(
                        ) == 0


class TestClassSuperFermionicOperatorsNoComplexPhase:
    nsite = 2
    fl_op = sf_op.SuperFermionicOperators(nsite,
                                          tilde_conjugationrule_phase=False)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=complex, format="csc")

    def test_commutation_relation_fock_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # purely fock creation and annihilation operators
                        anti_commutation_fock_c_cdag = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.cdag(j, s2))
                        anti_commutation_fock_c_c = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.c(j, s2))
                        anti_commutation_fock_cdag_cdag = \
                            fop.anti_commutator(
                                self.fl_op.cdag(i, s1),
                                self.fl_op.cdag(j, s2))

                        if i == j and s1 == s2:
                            assert (anti_commutation_fock_c_cdag -
                                    self.identity_super_fermionic).nnz == 0
                            assert anti_commutation_fock_c_c.nnz == 0
                            assert anti_commutation_fock_cdag_cdag.nnz == 0

                        else:
                            assert anti_commutation_fock_c_cdag.nnz == 0
                            assert anti_commutation_fock_c_c.nnz == 0
                            assert anti_commutation_fock_cdag_cdag.nnz == 0

    def test_commutation_relation_tilde_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # purely tilde creation and annihilation operators
                        anti_commutation_tilde_c_cdag = \
                            fop.anti_commutator(
                                self.fl_op.c_tilde(i, s1),
                                self.fl_op.cdag_tilde(j, s2))
                        anti_commutation_tilde_c_c =  \
                            fop.anti_commutator(
                                self.fl_op.c_tilde(i, s1),
                                self.fl_op.c_tilde(j, s2))
                        anti_commutation_tilde_cdag_cdag = \
                            fop.anti_commutator(
                                self.fl_op.cdag_tilde(i, s1),
                                self.fl_op.cdag_tilde(j, s2))

                        if i == j and s1 == s2:
                            assert (anti_commutation_tilde_c_cdag -
                                    self.identity_super_fermionic).nnz == 0
                            assert anti_commutation_tilde_c_c.nnz == 0
                            assert anti_commutation_tilde_cdag_cdag.nnz == 0

                        else:
                            assert anti_commutation_tilde_c_cdag.nnz == 0
                            assert anti_commutation_tilde_c_c.nnz == 0
                            assert anti_commutation_tilde_cdag_cdag.nnz == 0

    def test_commutation_relation_mixed_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # mixed creation and annihilation operators
                        anti_commutation_mixed_c_cdag_tilde = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.cdag_tilde(j, s2))
                        anti_commutation_mixed_c_c_tilde = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.c_tilde(j, s2))
                        anti_commutation_mixed_cdag_cdag_tilde = \
                            fop.anti_commutator(
                                self.fl_op.cdag(i, s1),
                                self.fl_op.cdag_tilde(j, s2))

                        if (not i == j) or (not s1 == s2):
                            assert anti_commutation_mixed_c_cdag_tilde.nnz == 0
                            assert anti_commutation_mixed_c_c_tilde.nnz == 0
                            assert \
                                anti_commutation_mixed_cdag_cdag_tilde.nnz == 0

    def test_fock_tilde_operator_correspondence(self):
        n_nonzeros = 0
        for i in range(self.nsite):
            for spin in ["up", "do"]:
                n_nonzeros += (self.fl_op.c(i, spin)
                               - self.fl_op.cdag_tilde(i, spin)
                               ).dot(self.fl_op.left_vacuum).nnz

                n_nonzeros += (self.fl_op.cdag(i, spin)
                               + self.fl_op.c_tilde(i, spin)
                               ).dot(self.fl_op.left_vacuum).nnz
        assert n_nonzeros == 0

    def test_fock_tilde_operator_correspondence_pair(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        cdag_c = (self.fl_op.cdag(i, s1)
                                  * self.fl_op.c(j, s2))

                        c_c = (self.fl_op.c(i, s1)
                               * self.fl_op.c(j, s2))

                        cdag_c_tilde = (
                            self.fl_op.cdag_tilde(j, s2)
                            * self.fl_op.c_tilde(i, s1))

                        c_c_tilde = (
                            self.fl_op.c_tilde(j, s2)
                            * self.fl_op.c_tilde(i, s1))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_pair_get_super_fermi_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        cdag_c = self.fl_op.get_super_fermionic_operator(
                            self.fl_op.fock_ops.cdag(i, s1)
                            * self.fl_op.fock_ops.c(j, s2))

                        c_c = self.fl_op.get_super_fermionic_operator(
                            self.fl_op.fock_ops.c(i, s1)
                            * self.fl_op.fock_ops.c(j, s2))

                        cdag_c_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(j, s2)
                                * self.fl_op.fock_ops.c(i, s1))

                        c_c_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.c(j, s2)
                                * self.fl_op.fock_ops.c(i, s1))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        for s1 in ["up", "do"]:
                            for s2 in ["up", "do"]:
                                for s3 in ["up", "do"]:
                                    for s4 in ["up", "do"]:
                                        cdag_cdag_c_c = (
                                            self.fl_op.cdag(i, s1)
                                            * self.fl_op.cdag(j, s1)
                                            * self.fl_op.c(k, s2)
                                            * self.fl_op.c(l, s2))

                                        cdag_c_cdag_c = (
                                            self.fl_op.cdag(i, s1)
                                            * self.fl_op.c(j, s1)
                                            * self.fl_op.cdag(k, s2)
                                            * self.fl_op.c(l, s2))

                                        cdag_c_c_cdag = (
                                            self.fl_op.cdag(i, s1)
                                            * self.fl_op.c(j, s1)
                                            * self.fl_op.c(k, s2)
                                            * self.fl_op.cdag(l, s2))

                                        cdag_cdag_c_c_tilde = (
                                            self.fl_op.cdag_tilde(l, s2)
                                            * self.fl_op.cdag_tilde(k, s2)
                                            * self.fl_op.c_tilde(j, s1)
                                            * self.fl_op.c_tilde(i, s1))

                                        cdag_c_cdag_c_tilde = (
                                            self.fl_op.cdag_tilde(l, s2)
                                            * self.fl_op.c_tilde(k, s2)
                                            * self.fl_op.cdag_tilde(j, s1)
                                            * self.fl_op.c_tilde(i, s1))

                                        cdag_c_c_cdag_tilde = (
                                            self.fl_op.c_tilde(l, s2)
                                            * self.fl_op.cdag_tilde(k, s2)
                                            * self.fl_op.cdag_tilde(j, s1)
                                            * self.fl_op.c_tilde(i, s1))

        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_cdag_c_c.count_nonzero() == 0

        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_cdag_c.count_nonzero() == 0

        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_c_cdag.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four_get_super_fermi_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        for s1 in ["up", "do"]:
                            for s2 in ["up", "do"]:
                                for s3 in ["up", "do"]:
                                    for s4 in ["up", "do"]:
                                        cdag_cdag_c_c = \
                                            (self.fl_op
                                             ).get_super_fermionic_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.cdag(
                                                    j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_cdag_c = \
                                            (self.fl_op
                                             ).get_super_fermionic_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.cdag(k,
                                                                           s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_c_cdag = \
                                            (self.fl_op
                                             ).get_super_fermionic_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.cdag(l,
                                                                           s2))

                                        cdag_cdag_c_c_tilde = \
                                            (self.fl_op
                                             ).get_super_fermionic_tilde_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.cdag(j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_cdag_c_tilde = \
                                            (self.fl_op
                                             ).get_super_fermionic_tilde_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.cdag(k,
                                                                           s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_c_cdag_tilde = \
                                            (self.fl_op
                                             ).get_super_fermionic_tilde_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.cdag(l,
                                                                           s2))

        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_cdag_c_c.count_nonzero() == 0

        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_cdag_c.count_nonzero() == 0

        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_c_cdag.count_nonzero() == 0


class TestClassSuperFermionicOperatorsComplexPhase:
    nsite = 2
    fl_op = sf_op.SuperFermionicOperators(nsite,
                                          tilde_conjugationrule_phase=True)

    identity_super_fermionic = sparse.eye(
        4**(fl_op.fock_ops.spin_times_site), dtype=complex, format="csc")

    def test_commutation_relation_fock_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # purely fock creation and annihilation operators
                        anti_commutation_fock_c_cdag = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.cdag(j, s2))
                        anti_commutation_fock_c_c = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.c(j, s2))
                        anti_commutation_fock_cdag_cdag = \
                            fop.anti_commutator(
                                self.fl_op.cdag(i, s1),
                                self.fl_op.cdag(j, s2))

                        if i == j and s1 == s2:
                            assert (anti_commutation_fock_c_cdag -
                                    self.identity_super_fermionic).nnz == 0
                            assert anti_commutation_fock_c_c.nnz == 0
                            assert anti_commutation_fock_cdag_cdag.nnz == 0

                        else:
                            assert anti_commutation_fock_c_cdag.nnz == 0
                            assert anti_commutation_fock_c_c.nnz == 0
                            assert anti_commutation_fock_cdag_cdag.nnz == 0

    def test_commutation_relation_tilde_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # purely tilde creation and annihilation operators
                        anti_commutation_tilde_c_cdag = \
                            fop.anti_commutator(
                                self.fl_op.c_tilde(i, s1),
                                self.fl_op.cdag_tilde(j, s2))
                        anti_commutation_tilde_c_c =  \
                            fop.anti_commutator(
                                self.fl_op.c_tilde(i, s1),
                                self.fl_op.c_tilde(j, s2))
                        anti_commutation_tilde_cdag_cdag = \
                            fop.anti_commutator(
                                self.fl_op.cdag_tilde(i, s1),
                                self.fl_op.cdag_tilde(j, s2))

                        if i == j and s1 == s2:
                            assert (anti_commutation_tilde_c_cdag -
                                    self.identity_super_fermionic).nnz == 0
                            assert anti_commutation_tilde_c_c.nnz == 0
                            assert anti_commutation_tilde_cdag_cdag.nnz == 0

                        else:
                            assert anti_commutation_tilde_c_cdag.nnz == 0
                            assert anti_commutation_tilde_c_c.nnz == 0
                            assert anti_commutation_tilde_cdag_cdag.nnz == 0

    def test_commutation_relation_mixed_subspace(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        # mixed creation and annihilation operators
                        anti_commutation_mixed_c_cdag_tilde = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.cdag_tilde(j, s2))
                        anti_commutation_mixed_c_c_tilde = \
                            fop.anti_commutator(self.fl_op.c(i, s1),
                                                self.fl_op.c_tilde(j, s2))
                        anti_commutation_mixed_cdag_cdag_tilde = \
                            fop.anti_commutator(
                                self.fl_op.cdag(i, s1), self.fl_op.cdag_tilde(
                                    j, s2))

                        if (not i == j) or (not s1 == s2):
                            assert anti_commutation_mixed_c_cdag_tilde.nnz == 0
                            assert anti_commutation_mixed_c_c_tilde.nnz == 0
                            assert \
                                anti_commutation_mixed_cdag_cdag_tilde.nnz == 0

    def test_fock_tilde_operator_correspondence(self):
        n_nonzeros = 0
        for i in range(self.nsite):
            for spin in ["up", "do"]:
                n_nonzeros += (self.fl_op.c(i, spin)
                               + 1j * self.fl_op.cdag_tilde(i, spin)
                               ).dot(self.fl_op.left_vacuum).nnz

                n_nonzeros += (self.fl_op.cdag(i, spin)
                               + 1j * self.fl_op.c_tilde(i, spin)
                               ).dot(self.fl_op.left_vacuum).nnz
        assert n_nonzeros == 0

    def test_fock_tilde_operator_correspondence_pair(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        cdag_c = (self.fl_op.cdag(i, s1)
                                  * self.fl_op.c(j, s2))

                        c_c = (self.fl_op.c(i, s1)
                               * self.fl_op.c(j, s2))

                        cdag_c_tilde = (
                            self.fl_op.cdag_tilde(j, s2)
                            * self.fl_op.c_tilde(i, s1))

                        c_c_tilde = (
                            self.fl_op.c_tilde(j, s2)
                            * self.fl_op.c_tilde(i, s1))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_pair_get_super_fermionic_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for s1 in ["up", "do"]:
                    for s2 in ["up", "do"]:
                        cdag_c = self.fl_op.get_super_fermionic_operator(
                            self.fl_op.fock_ops.cdag(i, s1)
                            * self.fl_op.fock_ops.c(j, s2))

                        c_c = self.fl_op.get_super_fermionic_operator(
                            self.fl_op.fock_ops.c(i, s1)
                            * self.fl_op.fock_ops.c(j, s2))

                        cdag_c_tilde = \
                            self.fl_op.get_super_fermionic_tilde_operator(
                                self.fl_op.fock_ops.cdag(j, s2)
                                * self.fl_op.fock_ops.c(i, s1))

                        c_c_tilde = self.fl_op.get_super_fermionic_tilde_operator(
                            self.fl_op.fock_ops.c(j, s2)
                            * self.fl_op.fock_ops.c(i, s1))

        correspondence_cdag_c = cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c.count_nonzero() == 0

        correspondence_c_c = c_c.dot(
            self.fl_op.left_vacuum) - \
            c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_c_c.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four(self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        for s1 in ["up", "do"]:
                            for s2 in ["up", "do"]:
                                for s3 in ["up", "do"]:
                                    for s4 in ["up", "do"]:
                                        cdag_cdag_c_c = (
                                            self.fl_op.cdag(i, s1)
                                            * self.fl_op.cdag(j, s1)
                                            * self.fl_op.c(k, s2)
                                            * self.fl_op.c(l, s2))

                                        cdag_c_cdag_c = (
                                            self.fl_op.cdag(i, s1)
                                            * self.fl_op.c(j, s1)
                                            * self.fl_op.cdag(k, s2)
                                            * self.fl_op.c(l, s2))

                                        cdag_c_c_cdag = (
                                            self.fl_op.cdag(i, s1)
                                            * self.fl_op.c(j, s1)
                                            * self.fl_op.c(k, s2)
                                            * self.fl_op.cdag(l, s2))

                                        cdag_cdag_c_c_tilde = (
                                            self.fl_op.cdag_tilde(l, s2)
                                            * self.fl_op.cdag_tilde(k, s2)
                                            * self.fl_op.c_tilde(j, s1)
                                            * self.fl_op.c_tilde(i, s1))

                                        cdag_c_cdag_c_tilde = (
                                            self.fl_op.cdag_tilde(l, s2)
                                            * self.fl_op.c_tilde(k, s2)
                                            * self.fl_op.cdag_tilde(j, s1)
                                            * self.fl_op.c_tilde(i, s1))

                                        cdag_c_c_cdag_tilde = (
                                            self.fl_op.c_tilde(l, s2)
                                            * self.fl_op.cdag_tilde(k, s2)
                                            * self.fl_op.cdag_tilde(j, s1)
                                            * self.fl_op.c_tilde(i, s1))

        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_cdag_c_c.count_nonzero() == 0

        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_cdag_c.count_nonzero() == 0

        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_c_cdag.count_nonzero() == 0

    def test_fock_tilde_operator_correspondence_four_get_super_fermionic_operator(
            self):
        for i in range(self.nsite):
            for j in range(self.nsite):
                for k in range(self.nsite):
                    for l in range(self.nsite):
                        for s1 in ["up", "do"]:
                            for s2 in ["up", "do"]:
                                for s3 in ["up", "do"]:
                                    for s4 in ["up", "do"]:
                                        cdag_cdag_c_c = \
                                            self.fl_op.get_super_fermionic_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.cdag(j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_cdag_c = \
                                            self.fl_op.get_super_fermionic_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.cdag(k, s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_c_cdag = \
                                            self.fl_op.get_super_fermionic_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.cdag(l, s2))

                                        cdag_cdag_c_c_tilde = \
                                            self.fl_op.get_super_fermionic_tilde_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.cdag(j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_cdag_c_tilde = \
                                            self.fl_op.get_super_fermionic_tilde_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.cdag(k, s2)
                                                * self.fl_op.fock_ops.c(l, s2))

                                        cdag_c_c_cdag_tilde = \
                                            self.fl_op.get_super_fermionic_tilde_operator(
                                                self.fl_op.fock_ops.cdag(i, s1)
                                                * self.fl_op.fock_ops.c(j, s1)
                                                * self.fl_op.fock_ops.c(k, s2)
                                                * self.fl_op.fock_ops.cdag(l, s2))

        correspondence_cdag_cdag_c_c = cdag_cdag_c_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_cdag_c_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_cdag_c_c.count_nonzero() == 0

        correspondence_cdag_c_cdag_c = cdag_c_cdag_c.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_cdag_c_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_cdag_c.count_nonzero() == 0

        correspondence_cdag_c_c_cdag = cdag_c_c_cdag.dot(
            self.fl_op.left_vacuum) - \
            cdag_c_c_cdag_tilde.dot(self.fl_op.left_vacuum)
        assert correspondence_cdag_c_c_cdag.count_nonzero() == 0


if __name__ == "__main__":

    pytest.main("-v test_define_super_fermionic_operators.py")
